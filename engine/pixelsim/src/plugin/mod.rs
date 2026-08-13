use bevy::diagnostic::{Diagnostic, RegisterDiagnostic};
use common::prelude::*;

mod messages;
pub use messages::*;

mod resources;
pub use resources::*;

mod systems;
pub use systems::*;

use crate::units::TICK_HZ;

pub struct PixelSimPlugin {
    pub config: SimConfig,
}

impl Default for PixelSimPlugin {
    fn default() -> Self {
        Self {
            config: SimConfig::default(),
        }
    }
}

impl Plugin for PixelSimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Time::<Fixed>::from_hz(TICK_HZ))
            .insert_resource(self.config.clone())
            // Placeholder only: `build_layers` moves this to the Plant once the stack
            // exists, since which index that is depends on the configured slots.
            .insert_resource(ActiveLayer(0))
            .insert_resource(Journal(crate::undo::UndoJournal::new()))
            .insert_resource(Stats(crate::stats::SimStats::default()))
            .insert_resource(Paused(false))
            .insert_resource(StepOnce(false))
            .add_message::<LayerEdited>()
            .add_message::<PortTransferred>()
            .register_diagnostic(Diagnostic::new(DIAG_STEP_MS).with_suffix("ms"))
            .register_diagnostic(Diagnostic::new(DIAG_ACTIVE_CELLS))
            .register_diagnostic(Diagnostic::new(DIAG_AWAKE_CHUNKS))
            .add_systems(Startup, (load_materials, build_layers).chain())
            .add_systems(
                FixedUpdate,
                (step_sim, step_coarse, publish_diagnostics).chain(),
            );
    }
}
