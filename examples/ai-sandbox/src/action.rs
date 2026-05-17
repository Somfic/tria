use crate::state::AiState;
use goap::{Action, Effect, PlanContext, Precondition, WorldState};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum AiAction {
    MoveToGun,
    PickupGun,
    MoveToAmmo,
    PickupAmmo,
    LoadGun,
    MoveToCover,
    EnterCover,
    Shoot,
    MoveToMedkit,
    UseMedkit,
}

impl Action<AiState> for AiAction {
    fn preconditions(&self) -> Precondition<AiState> {
        use AiAction::*;
        use AiState::*;
        match self {
            MoveToGun => Precondition::new(),
            PickupGun => Precondition::new()
                .requires(NearGun, true)
                .requires(HasGun, false),
            MoveToAmmo => Precondition::new(),
            PickupAmmo => Precondition::new()
                .requires(NearAmmo, true)
                .requires(HasAmmo, false),
            LoadGun => Precondition::new()
                .requires(HasGun, true)
                .requires(HasAmmo, true)
                .requires(GunLoaded, false),
            MoveToCover => Precondition::new(),
            EnterCover => Precondition::new()
                .requires(NearCover, true)
                .requires(InCover, false),
            Shoot => Precondition::new()
                .requires(HasGun, true)
                .requires(GunLoaded, true)
                .requires(InCover, true),
            MoveToMedkit => Precondition::new(),
            UseMedkit => Precondition::new()
                .requires(NearMedkit, true)
                .requires(Hurt, true),
        }
    }

    fn effects(&self) -> Effect<AiState> {
        use AiAction::*;
        use AiState::*;
        match self {
            MoveToGun => Effect::new()
                .sets(NearGun, true)
                .sets(NearAmmo, false)
                .sets(NearCover, false)
                .sets(NearMedkit, false)
                .sets(InCover, false),
            PickupGun => Effect::new().sets(HasGun, true),
            MoveToAmmo => Effect::new()
                .sets(NearAmmo, true)
                .sets(NearGun, false)
                .sets(NearCover, false)
                .sets(NearMedkit, false)
                .sets(InCover, false),
            PickupAmmo => Effect::new().sets(HasAmmo, true),
            LoadGun => Effect::new().sets(GunLoaded, true).sets(HasAmmo, false),
            MoveToCover => Effect::new()
                .sets(NearCover, true)
                .sets(NearGun, false)
                .sets(NearAmmo, false)
                .sets(NearMedkit, false),
            EnterCover => Effect::new().sets(InCover, true),
            Shoot => Effect::new().sets(TargetDead, true),
            MoveToMedkit => Effect::new()
                .sets(NearMedkit, true)
                .sets(NearGun, false)
                .sets(NearAmmo, false)
                .sets(NearCover, false)
                .sets(InCover, false),
            UseMedkit => Effect::new().sets(Hurt, false),
        }
    }

    fn cost(&self, _state: &WorldState<AiState>, _ctx: &PlanContext) -> f32 {
        1.0
    }
}
