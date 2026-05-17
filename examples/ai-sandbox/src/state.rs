use common::prelude::*;
use goap::StateKey;

pub const MAX_HEALTH: f32 = 100.0;
pub const HURT_THRESHOLD: f32 = 40.0;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum AiState {
    HasGun = 0,
    HasAmmo = 1,
    GunLoaded = 2,
    InCover = 3,
    NearGun = 4,
    NearAmmo = 5,
    NearCover = 6,
    TargetDead = 7,
    Hurt = 8,
    NearMedkit = 9,
}

impl StateKey for AiState {
    fn bit_position(self) -> u8 {
        self as u8
    }
    fn all() -> &'static [Self] {
        use AiState::*;
        &[
            HasGun, HasAmmo, GunLoaded, InCover, NearGun, NearAmmo, NearCover, TargetDead, Hurt,
            NearMedkit,
        ]
    }
}

/// Game-side NPC state: what the NPC physically has/is, separate from
/// what the agent *perceives*. The two are usually in sync but the
/// distinction keeps the handler logic clean.
#[derive(Component)]
pub struct NpcState {
    pub has_gun: bool,
    /// A picked-up clip that hasn't been loaded into the gun yet.
    pub has_reserve_ammo: bool,
    /// Rounds currently in the gun. `> 0` means the gun is loaded.
    pub loaded_rounds: u32,
    pub in_cover: bool,
    pub health: f32,
}

impl Default for NpcState {
    fn default() -> Self {
        Self {
            has_gun: false,
            has_reserve_ammo: false,
            loaded_rounds: 0,
            in_cover: false,
            health: MAX_HEALTH,
        }
    }
}
