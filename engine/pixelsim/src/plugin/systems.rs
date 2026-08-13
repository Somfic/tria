use bevy::diagnostic::{DiagnosticPath, Diagnostics};
use common::prelude::*;

use super::resources::{ActiveLayer, Materials, Paused, Repose, Sim, SimConfig, Stats, StepOnce};
use crate::coarse::GasPlane;
use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::repose::ReposeLut;
use crate::step::{step_all, step_all_serial};
use crate::units::CALIBRATE_GRAINS;

pub const DIAG_STEP_MS: DiagnosticPath = DiagnosticPath::const_new("sim/step_ms");
pub const DIAG_ACTIVE_CELLS: DiagnosticPath = DiagnosticPath::const_new("sim/active_cells");
pub const DIAG_AWAKE_CHUNKS: DiagnosticPath = DiagnosticPath::const_new("sim/awake_chunks");

pub fn load_materials(mut commands: Commands, config: Res<SimConfig>) {
    let table = match &config.material_path {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(src) => match MaterialTable::from_toml_str(&src) {
                Ok(t) => t,
                Err(e) => {
                    error!("{}: {e}; using the embedded material table", path.display());
                    MaterialTable::embedded()
                }
            },
            Err(e) => {
                error!("{}: {e}; using the embedded material table", path.display());
                MaterialTable::embedded()
            }
        },
        None => MaterialTable::embedded(),
    };

    let mut lut = ReposeLut::analytic();
    if config.calibrate_repose {
        let angles: Vec<f32> = table
            .iter()
            .filter(|(_, m)| m.class == MaterialClass::Powder)
            .map(|(_, m)| m.repose_angle)
            .collect();
        let t0 = std::time::Instant::now();
        lut.calibrate(&table, &angles, CALIBRATE_GRAINS, config.seed);
        info!(
            "repose LUT calibrated for {} powder angle(s) in {:.0} ms",
            angles.len(),
            t0.elapsed().as_secs_f32() * 1000.0
        );
    }

    commands.insert_resource(Materials(table));
    commands.insert_resource(Repose(lut));
}

/// Builds one [`Layer`] per configured slot, front (Face) to back (Hold), plus the
/// single layer-agnostic gas plane. Deliberately does not read `Materials`: it only
/// needs geometry, so it cannot depend on a command from the previous system having
/// been applied.
pub fn build_layers(mut commands: Commands, config: Res<SimConfig>) {
    // `One Simulated Plane` is a design invariant, so it is enforced here rather than
    // trusted. Two simulated slabs would step twice, cost twice and — worse — quietly work,
    // which is how the old six-slab stack survived as long as it did.
    let simulated = config.slots.iter().filter(|s| s.simulated()).count();
    assert!(
        simulated <= 1,
        "{simulated} simulated slabs in the stack {:?}; exactly one slab may be `Plant` \
         (see the vault's `One Simulated Plane`). Deck/Works/Hold are vertical floors \
         *inside* the Plant now, not separate slots.",
        config.slots
    );

    let layers: Vec<Layer> = config
        .slots
        .iter()
        .enumerate()
        .map(|(i, &slot)| {
            let seed = config.seed.wrapping_mul(1) ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            Layer::new(config.width, config.height, slot, seed)
        })
        .collect();

    // Start in the Plant, not in whatever slab happens to be index 0. The active slab is
    // where sculpting lands and what the camera is dollied to keep pixel-exact, so starting
    // on the thin decorative Face slab means the player opens the game unable to touch
    // anything that matters.
    let plant = layers
        .iter()
        .position(|l| l.slot.simulated())
        .unwrap_or(0);
    commands.insert_resource(ActiveLayer(plant));

    commands.insert_resource(Sim {
        layers,
        ports: Vec::new(),
        gas: GasPlane::new(config.width, config.height),
    });
}

pub fn step_sim(
    mut sim: ResMut<Sim>,
    materials: Res<Materials>,
    repose: Res<Repose>,
    config: Res<SimConfig>,
    mut stats: ResMut<Stats>,
    paused: Res<Paused>,
    mut step_once: ResMut<StepOnce>,
) {
    if paused.0 && !step_once.0 {
        return;
    }
    step_once.0 = false;
    if sim.layers.is_empty() {
        return;
    }

    let tick = sim.layers.iter().map(|l| l.tick).max().unwrap_or(0) + 1;
    let sim = &mut *sim;
    let (layers, ports) = (&mut sim.layers, &sim.ports);

    let pool = if config.parallel {
        bevy::tasks::ComputeTaskPool::try_get()
    } else {
        None
    };
    stats.0 = match pool {
        Some(pool) => step_all(layers, ports, &materials.0, &repose.0, tick, pool),
        None => step_all_serial(layers, ports, &materials.0, &repose.0, tick),
    };
}

/// Coarse-grid hook point. `flow_mag`/`activity` are written and decayed inside
/// `step_layer`; `heat`, `pressure` and the gas plane are allocated and untouched in
/// the slice, and this is the system that will step them.
pub fn step_coarse(sim: Res<Sim>, config: Res<SimConfig>) {
    let _ = (&sim, &config);
}

pub fn publish_diagnostics(
    mut diagnostics: Diagnostics,
    stats: Res<Stats>,
    active: Res<ActiveLayer>,
) {
    let _ = &active;
    diagnostics.add_measurement(&DIAG_STEP_MS, || stats.0.step_ms as f64);
    diagnostics.add_measurement(&DIAG_ACTIVE_CELLS, || stats.0.active_cells as f64);
    diagnostics.add_measurement(&DIAG_AWAKE_CHUNKS, || stats.0.awake_chunks as f64);
}
