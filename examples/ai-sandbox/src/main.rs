mod action;
mod handlers;
mod perception;
mod state;

use std::time::{SystemTime, UNIX_EPOCH};

use common::prelude::*;
use goap::{
    ActionSet, CurrentGoal, CurrentPlan, Goal, GoapAgent, GoapPlugin, PerceivedState, WorldState,
    advance_plan, execute_plan, give_plan,
};

use action::AiAction;
use state::{AiState, HURT_THRESHOLD, NpcState};

#[derive(Component)]
pub struct WorldGun;

#[derive(Component)]
pub struct WorldAmmo;

#[derive(Component)]
pub struct CoverSpot;

#[derive(Component)]
pub struct Medkit;

#[derive(Component)]
pub struct Target;

#[derive(Component)]
pub struct HudText;

#[derive(Resource, Default)]
pub struct Score {
    pub kills: u32,
}

#[derive(Resource)]
pub struct Rng(u64);

impl Rng {
    fn seeded() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self(seed | 1) // never zero
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    pub fn pos(&mut self) -> Vec3 {
        Vec3::new(self.range(-520.0, 520.0), self.range(-300.0, 300.0), 0.0)
    }
}

pub fn spawn_target(commands: &mut Commands, pos: Vec3) {
    commands.spawn((
        Target,
        Sprite {
            color: Color::srgb(0.9, 0.2, 0.2),
            custom_size: Some(Vec2::splat(30.0)),
            ..default()
        },
        Transform::from_translation(pos),
    ));
}

pub fn spawn_ammo(commands: &mut Commands, pos: Vec3) {
    commands.spawn((
        WorldAmmo,
        Sprite {
            color: Color::srgb(0.9, 0.7, 0.2),
            custom_size: Some(Vec2::splat(20.0)),
            ..default()
        },
        Transform::from_translation(pos),
    ));
}

pub fn spawn_medkit(commands: &mut Commands, pos: Vec3) {
    commands.spawn((
        Medkit,
        Sprite {
            color: Color::srgb(0.4, 1.0, 0.6),
            custom_size: Some(Vec2::splat(22.0)),
            ..default()
        },
        Transform::from_translation(pos),
    ));
}

fn kill_goal() -> Goal<AiState> {
    Goal::new("kill_target").wants(AiState::TargetDead, true)
}

fn survive_goal() -> Goal<AiState> {
    Goal::new("survive")
        .wants(AiState::Hurt, false)
        .with_priority(10.0)
}

fn setup_scene(mut commands: Commands, mut rng: ResMut<Rng>) {
    commands.spawn(Camera2d);

    commands.spawn((
        WorldGun,
        Sprite {
            color: Color::srgb(0.3, 0.3, 0.9),
            custom_size: Some(Vec2::splat(20.0)),
            ..default()
        },
        Transform::from_xyz(-200.0, 100.0, 0.0),
    ));
    for (x, y) in [
        (-380.0, -180.0),
        (-120.0, -260.0),
        (160.0, -200.0),
        (400.0, -60.0),
        (-260.0, 160.0),
        (220.0, 220.0),
    ] {
        commands.spawn((
            CoverSpot,
            Sprite {
                color: Color::srgb(0.4, 0.4, 0.4),
                custom_size: Some(Vec2::splat(40.0)),
                ..default()
            },
            Transform::from_xyz(x, y, 0.0),
        ));
    }

    let p = rng.pos();
    spawn_ammo(&mut commands, p);
    let p = rng.pos();
    spawn_medkit(&mut commands, p);
    let p = rng.pos();
    spawn_target(&mut commands, p);

    let actions = ActionSet::new(vec![
        AiAction::MoveToGun,
        AiAction::PickupGun,
        AiAction::MoveToAmmo,
        AiAction::PickupAmmo,
        AiAction::LoadGun,
        AiAction::MoveToCover,
        AiAction::EnterCover,
        AiAction::Shoot,
        AiAction::MoveToMedkit,
        AiAction::UseMedkit,
    ]);

    commands.spawn((
        GoapAgent,
        PerceivedState::<AiState>(WorldState::new()),
        actions,
        CurrentGoal(kill_goal()),
        NpcState::default(),
        Sprite {
            color: Color::srgb(0.2, 0.9, 0.3),
            custom_size: Some(Vec2::splat(25.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    commands.spawn((
        HudText,
        Text::new("..."),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        },
    ));
}

fn select_goal(
    mut commands: Commands,
    mut agents: Query<(Entity, &NpcState, &mut CurrentGoal<AiState>), With<GoapAgent>>,
) {
    for (entity, state, mut goal) in &mut agents {
        let want_survive = state.health < HURT_THRESHOLD;
        let desired = if want_survive {
            "survive"
        } else {
            "kill_target"
        };

        if goal.0.name() != desired {
            goal.0 = if want_survive {
                survive_goal()
            } else {
                kill_goal()
            };
            commands
                .entity(entity)
                .remove::<CurrentPlan<AiState, AiAction>>();
            info!("goal -> {}", desired);
        }
    }
}

fn update_hud(
    score: Res<Score>,
    npc: Query<(&NpcState, &CurrentGoal<AiState>), With<GoapAgent>>,
    mut hud: Query<&mut Text, With<HudText>>,
) {
    let Some((state, goal)) = npc.iter().next() else {
        return;
    };
    let Some(mut text) = hud.iter_mut().next() else {
        return;
    };
    *text = Text::new(format!(
        "HP: {:.0}   Goal: {}   Loaded: {}   Reserve: {}   Kills: {}",
        state.health,
        goal.0.name(),
        state.loaded_rounds,
        if state.has_reserve_ammo { "yes" } else { "no" },
        score.kills,
    ));
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(GoapPlugin::<AiState, AiAction>::default())
        .insert_resource(Rng::seeded())
        .init_resource::<Score>()
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                perception::update_perception.before(give_plan::<AiState, AiAction>),
                select_goal.before(give_plan::<AiState, AiAction>),
                handlers::handle_actions
                    .after(execute_plan::<AiState, AiAction>)
                    .before(advance_plan::<AiState, AiAction>),
                handlers::walk_system.before(advance_plan::<AiState, AiAction>),
                handlers::shoot_system.before(advance_plan::<AiState, AiAction>),
                handlers::bullet_system,
                handlers::enemy_fire,
                handlers::enemy_bullet_system,
                update_hud,
            ),
        )
        .run();
}
