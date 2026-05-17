use common::prelude::*;
use goap::{ActionCompleted, ActionStartRequested, GoapAgent};

use crate::action::AiAction;
use crate::state::{AiState, MAX_HEALTH, NpcState};
use crate::{
    CoverSpot, Medkit, Rng, Score, Target, WorldAmmo, WorldGun, spawn_ammo, spawn_medkit,
    spawn_target,
};

const WALK_SPEED: f32 = 140.0; // px / second
const BULLET_SPEED: f32 = 700.0; // px / second
const ARRIVE_EPS: f32 = 2.0;

const CLIP_SIZE: u32 = 5; // rounds per loaded clip
const FIRE_TIME: f32 = 0.45; // seconds to aim+fire one shot

const MISS_CHANCE: f32 = 0.35; // fraction of shots that go wide
const TRUE_SPREAD: f32 = 0.015; // rad: jitter on a well-aimed shot
const MISS_SPREAD: (f32, f32) = (0.20, 0.50); // rad: deviation on a wide shot

const BULLET_HIT_RADIUS: f32 = 20.0;
const BULLET_MAX_RANGE: f32 = 1600.0;

const COVER_BLOCK_RADIUS: f32 = 24.0;
const STANCE_BEHIND: f32 = 26.0;
const MUZZLE_FWD: f32 = 18.0;
const PEEK: f32 = 30.0;

const ENEMY_FIRE_INTERVAL: f32 = 1.3; // seconds between volleys
const ENEMY_BULLET_SPEED: f32 = 430.0;
const ENEMY_DMG: f32 = 16.0;
const ENEMY_HIT_RADIUS: f32 = 16.0;
const ENEMY_BULLET_RANGE: f32 = 1400.0;
const ENEMY_SPREAD: f32 = 0.11; // rad

#[derive(Component)]
pub struct Walking {
    dest: Vec3,
    action: AiAction,
}

#[derive(Component)]
pub struct Shooting {
    timer: f32,
    dir: Vec3,
    muzzle: Vec3,
}

#[derive(Component)]
pub struct Bullet {
    vel: Vec3,
    traveled: f32,
}

#[derive(Component)]
pub struct EnemyBullet {
    vel: Vec3,
    traveled: f32,
}

pub fn handle_actions(
    mut reader: MessageReader<ActionStartRequested<AiState, AiAction>>,
    mut writer: MessageWriter<ActionCompleted<AiState, AiAction>>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
    mut agents: Query<(&Transform, &mut NpcState), With<GoapAgent>>,
    guns: Query<(Entity, &Transform), (With<WorldGun>, Without<GoapAgent>)>,
    ammo: Query<(Entity, &Transform), (With<WorldAmmo>, Without<GoapAgent>)>,
    cover: Query<&Transform, (With<CoverSpot>, Without<GoapAgent>)>,
    medkits: Query<(Entity, &Transform), (With<Medkit>, Without<GoapAgent>)>,
    target: Query<(Entity, &Transform), (With<Target>, Without<GoapAgent>)>,
) {
    for msg in reader.read() {
        let Ok((npc_tf, mut npc_state)) = agents.get_mut(msg.agent) else {
            continue;
        };
        let here = npc_tf.translation;

        let walk_to = |commands: &mut Commands,
                       writer: &mut MessageWriter<ActionCompleted<AiState, AiAction>>,
                       dest: Option<Vec3>| match dest {
            Some(dest) => {
                commands.entity(msg.agent).insert(Walking {
                    dest,
                    action: msg.action,
                });
            }
            None => {
                writer.write(ActionCompleted::new(msg.agent, msg.action, false));
            }
        };

        let nearest = |positions: &mut dyn Iterator<Item = Vec3>| {
            positions.min_by(|a, b| a.distance(here).total_cmp(&b.distance(here)))
        };

        match msg.action {
            AiAction::MoveToGun => {
                npc_state.in_cover = false; // leaving cover
                walk_to(
                    &mut commands,
                    &mut writer,
                    guns.iter().next().map(|(_, t)| t.translation),
                )
            }
            AiAction::MoveToAmmo => {
                npc_state.in_cover = false; // leaving cover
                walk_to(
                    &mut commands,
                    &mut writer,
                    ammo.iter().next().map(|(_, t)| t.translation),
                )
            }
            AiAction::MoveToMedkit => {
                npc_state.in_cover = false; // leaving cover
                let dest = nearest(&mut medkits.iter().map(|(_, t)| t.translation));
                walk_to(&mut commands, &mut writer, dest)
            }
            AiAction::MoveToCover => {
                let cover_pos = nearest(&mut cover.iter().map(|t| t.translation));
                let dest = cover_pos.map(|c| match target.iter().next() {
                    Some((_, t)) => {
                        let mut dir = (t.translation - c).truncate().normalize_or_zero();
                        if dir == Vec2::ZERO {
                            dir = Vec2::Y;
                        }
                        c - Vec3::new(dir.x, dir.y, 0.0) * STANCE_BEHIND
                    }
                    None => c,
                });
                walk_to(&mut commands, &mut writer, dest)
            }

            AiAction::PickupGun => {
                if let Some((e, _)) = guns.iter().next() {
                    commands.entity(e).despawn();
                    npc_state.has_gun = true;
                    writer.write(ActionCompleted::new(msg.agent, msg.action, true));
                } else {
                    writer.write(ActionCompleted::new(msg.agent, msg.action, false));
                }
            }
            AiAction::PickupAmmo => {
                if let Some((e, _)) = ammo.iter().next() {
                    commands.entity(e).despawn();
                    npc_state.has_reserve_ammo = true;
                    spawn_ammo(&mut commands, rng.pos());
                    writer.write(ActionCompleted::new(msg.agent, msg.action, true));
                } else {
                    writer.write(ActionCompleted::new(msg.agent, msg.action, false));
                }
            }
            AiAction::LoadGun => {
                npc_state.loaded_rounds = CLIP_SIZE;
                npc_state.has_reserve_ammo = false;
                writer.write(ActionCompleted::new(msg.agent, msg.action, true));
            }
            AiAction::EnterCover => {
                npc_state.in_cover = true;
                writer.write(ActionCompleted::new(msg.agent, msg.action, true));
            }
            AiAction::UseMedkit => {
                if let Some((e, _)) = medkits.iter().next() {
                    commands.entity(e).despawn();
                    npc_state.health = MAX_HEALTH;
                    spawn_medkit(&mut commands, rng.pos());
                    info!("patched up, health = {}", npc_state.health);
                    writer.write(ActionCompleted::new(msg.agent, msg.action, true));
                } else {
                    writer.write(ActionCompleted::new(msg.agent, msg.action, false));
                }
            }

            AiAction::Shoot => {
                if npc_state.loaded_rounds == 0 {
                    writer.write(ActionCompleted::new(msg.agent, msg.action, false));
                } else if let Some((_, target_tf)) = target.iter().next() {
                    let tgt = target_tf.translation;

                    // lean-out muzzle: to the side of and ahead of the
                    // body so the NPC's own cover can't block the shot
                    let mut body_dir = (tgt - here).truncate().normalize_or_zero();
                    if body_dir == Vec2::ZERO {
                        body_dir = Vec2::X;
                    }
                    let perp = Vec3::new(-body_dir.y, body_dir.x, 0.0);
                    let muzzle =
                        here + Vec3::new(body_dir.x, body_dir.y, 0.0) * MUZZLE_FWD + perp * PEEK;

                    // aim from the muzzle, not the body, so the lean
                    // offset doesn't shift the shot off-target
                    let to_target = (tgt - muzzle).truncate();
                    let wide = rng.f32() < MISS_CHANCE;
                    let offset = if wide {
                        let sign = if rng.f32() < 0.5 { -1.0 } else { 1.0 };
                        sign * rng.range(MISS_SPREAD.0, MISS_SPREAD.1)
                    } else {
                        rng.range(-TRUE_SPREAD, TRUE_SPREAD)
                    };
                    let angle = to_target.to_angle() + offset;
                    commands.entity(msg.agent).insert(Shooting {
                        timer: FIRE_TIME,
                        dir: Vec3::new(angle.cos(), angle.sin(), 0.0),
                        muzzle: muzzle.with_z(1.0),
                    });
                } else {
                    writer.write(ActionCompleted::new(msg.agent, msg.action, false));
                }
            }
        }

        info!("action {:?} started", msg.action);
    }
}

pub fn walk_system(
    time: Res<Time>,
    mut commands: Commands,
    mut writer: MessageWriter<ActionCompleted<AiState, AiAction>>,
    mut walkers: Query<(Entity, &mut Transform, &Walking)>,
) {
    for (e, mut tf, walking) in &mut walkers {
        let to = walking.dest - tf.translation;
        let dist = to.length();
        let step = WALK_SPEED * time.delta_secs();

        if dist <= step.max(ARRIVE_EPS) {
            tf.translation = walking.dest;
            commands.entity(e).remove::<Walking>();
            info!("arrived, {:?} complete", walking.action);
            writer.write(ActionCompleted::new(e, walking.action, true));
        } else {
            tf.translation += to / dist * step;
        }
    }
}

pub fn shoot_system(
    time: Res<Time>,
    mut commands: Commands,
    mut writer: MessageWriter<ActionCompleted<AiState, AiAction>>,
    mut shooters: Query<(Entity, &mut NpcState, &mut Shooting)>,
) {
    for (e, mut npc_state, mut shooting) in &mut shooters {
        shooting.timer -= time.delta_secs();
        if shooting.timer > 0.0 {
            continue;
        }

        commands.entity(e).remove::<Shooting>();

        if npc_state.loaded_rounds > 0 {
            npc_state.loaded_rounds -= 1;
            commands.spawn((
                Bullet {
                    vel: shooting.dir * BULLET_SPEED,
                    traveled: 0.0,
                },
                Sprite {
                    color: Color::srgb(1.0, 1.0, 0.4),
                    custom_size: Some(Vec2::splat(6.0)),
                    ..default()
                },
                Transform::from_translation(shooting.muzzle),
            ));
        }
        info!("shot fired, {} rounds left", npc_state.loaded_rounds);
        writer.write(ActionCompleted::new(e, AiAction::Shoot, true));
    }
}

pub fn bullet_system(
    time: Res<Time>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
    mut score: ResMut<Score>,
    mut bullets: Query<(Entity, &mut Transform, &mut Bullet)>,
    targets: Query<(Entity, &Transform), (With<Target>, Without<Bullet>)>,
    cover: Query<&Transform, (With<CoverSpot>, Without<Bullet>)>,
) {
    for (be, mut btf, mut bullet) in &mut bullets {
        let step = bullet.vel * time.delta_secs();
        btf.translation += step;
        bullet.traveled += step.length();

        if cover
            .iter()
            .any(|c| c.translation.distance(btf.translation) < COVER_BLOCK_RADIUS)
        {
            commands.entity(be).try_despawn();
            continue;
        }

        let hit = targets
            .iter()
            .find(|(_, t)| t.translation.distance(btf.translation) < BULLET_HIT_RADIUS)
            .map(|(e, _)| e);

        if let Some(target_e) = hit {
            commands.entity(target_e).try_despawn();
            spawn_target(&mut commands, rng.pos());
            commands.entity(be).try_despawn();
            score.kills += 1;
            info!("hit! kills = {}", score.kills);
        } else if bullet.traveled > BULLET_MAX_RANGE {
            commands.entity(be).try_despawn(); // miss
        }
    }
}

pub fn enemy_fire(
    time: Res<Time>,
    mut cooldown: Local<f32>,
    mut commands: Commands,
    mut rng: ResMut<Rng>,
    npc: Query<&Transform, With<GoapAgent>>,
    targets: Query<&Transform, (With<Target>, Without<GoapAgent>)>,
) {
    *cooldown -= time.delta_secs();
    if *cooldown > 0.0 {
        return;
    }
    *cooldown = ENEMY_FIRE_INTERVAL;

    let Some(npc_tf) = npc.iter().next() else {
        return;
    };

    for target_tf in &targets {
        let aim = (npc_tf.translation - target_tf.translation).truncate();
        let angle = aim.to_angle() + rng.range(-ENEMY_SPREAD, ENEMY_SPREAD);
        commands.spawn((
            EnemyBullet {
                vel: Vec3::new(angle.cos(), angle.sin(), 0.0) * ENEMY_BULLET_SPEED,
                traveled: 0.0,
            },
            Sprite {
                color: Color::srgb(1.0, 0.4, 0.2),
                custom_size: Some(Vec2::splat(6.0)),
                ..default()
            },
            Transform::from_translation(target_tf.translation.with_z(1.0)),
        ));
    }
}

pub fn enemy_bullet_system(
    time: Res<Time>,
    mut commands: Commands,
    mut bullets: Query<(Entity, &mut Transform, &mut EnemyBullet)>,
    mut npc: Query<(&Transform, &mut NpcState), (With<GoapAgent>, Without<EnemyBullet>)>,
    cover: Query<&Transform, (With<CoverSpot>, Without<EnemyBullet>)>,
) {
    let Some((npc_tf, mut npc_state)) = npc.iter_mut().next() else {
        return;
    };

    for (be, mut btf, mut bullet) in &mut bullets {
        let step = bullet.vel * time.delta_secs();
        btf.translation += step;
        bullet.traveled += step.length();

        if cover
            .iter()
            .any(|c| c.translation.distance(btf.translation) < COVER_BLOCK_RADIUS)
        {
            commands.entity(be).try_despawn();
            continue;
        }

        if btf.translation.distance(npc_tf.translation) < ENEMY_HIT_RADIUS {
            npc_state.health = (npc_state.health - ENEMY_DMG).max(0.0);
            commands.entity(be).try_despawn();
            info!("NPC hit, health = {}", npc_state.health);
        } else if bullet.traveled > ENEMY_BULLET_RANGE {
            commands.entity(be).try_despawn();
        }
    }
}
