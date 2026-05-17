use common::prelude::*;
use goap::{GoapAgent, PerceivedState};

use crate::state::{AiState, HURT_THRESHOLD, NpcState};
use crate::{CoverSpot, Medkit, Target, WorldAmmo, WorldGun};

const NEAR_DISTANCE: f32 = 30.0;

pub fn update_perception(
    mut agents: Query<(&Transform, &NpcState, &mut PerceivedState<AiState>), With<GoapAgent>>,
    guns: Query<&Transform, (With<WorldGun>, Without<GoapAgent>)>,
    ammo: Query<&Transform, (With<WorldAmmo>, Without<GoapAgent>)>,
    cover: Query<&Transform, (With<CoverSpot>, Without<GoapAgent>)>,
    medkit: Query<&Transform, (With<Medkit>, Without<GoapAgent>)>,
    target: Query<&Transform, (With<Target>, Without<GoapAgent>)>,
) {
    for (npc_tf, npc_state, mut perceived) in &mut agents {
        let pos = npc_tf.translation;
        let near_gun = guns
            .iter()
            .any(|t| t.translation.distance(pos) < NEAR_DISTANCE);
        let near_ammo = ammo
            .iter()
            .any(|t| t.translation.distance(pos) < NEAR_DISTANCE);
        let near_cover = cover
            .iter()
            .any(|t| t.translation.distance(pos) < NEAR_DISTANCE);
        let near_medkit = medkit
            .iter()
            .any(|t| t.translation.distance(pos) < NEAR_DISTANCE);

        perceived.0.set(AiState::NearGun, near_gun);
        perceived.0.set(AiState::NearAmmo, near_ammo);
        perceived.0.set(AiState::NearCover, near_cover);
        perceived.0.set(AiState::NearMedkit, near_medkit);
        perceived
            .0
            .set(AiState::TargetDead, target.iter().count() == 0);

        perceived.0.set(AiState::HasGun, npc_state.has_gun);
        perceived
            .0
            .set(AiState::HasAmmo, npc_state.has_reserve_ammo);
        perceived
            .0
            .set(AiState::GunLoaded, npc_state.loaded_rounds > 0);
        perceived.0.set(AiState::InCover, npc_state.in_cover);
        perceived
            .0
            .set(AiState::Hurt, npc_state.health < HURT_THRESHOLD);
    }
}
