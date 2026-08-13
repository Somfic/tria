use common::prelude::*;

use crate::coarse::GasPlane;
use crate::layer::Layer;
use crate::material::MaterialTable;
use crate::port::PortPair;
use crate::repose::ReposeLut;
use crate::stats::SimStats;
use crate::undo::UndoJournal;
use crate::units::LayerSlot;

#[derive(Resource)]
pub struct Sim {
    pub layers: Vec<Layer>,
    pub ports: Vec<PortPair>,
    pub gas: GasPlane,
}

impl Sim {
    pub fn layer(&self, i: usize) -> &Layer {
        &self.layers[i]
    }

    pub fn layer_mut(&mut self, i: usize) -> &mut Layer {
        &mut self.layers[i]
    }

    /// for undo / ports only
    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    pub fn index_of(&self, slot: LayerSlot) -> Option<usize> {
        self.layers.iter().position(|l| l.slot == slot)
    }

    /// signed: negative = in front of the active layer
    pub fn depth_from(&self, active: usize, i: usize) -> i32 {
        i as i32 - active as i32
    }

    pub fn add_port(&mut self, p: PortPair) {
        self.ports.push(p);
    }
}

#[derive(Resource)]
pub struct Materials(pub MaterialTable);

#[derive(Resource)]
pub struct Repose(pub ReposeLut);

#[derive(Resource)]
pub struct ActiveLayer(pub usize);

#[derive(Resource)]
pub struct Journal(pub UndoJournal);

#[derive(Resource)]
pub struct Stats(pub SimStats);

#[derive(Resource)]
pub struct Paused(pub bool);

#[derive(Resource)]
pub struct StepOnce(pub bool);

#[derive(Resource, Clone, Debug)]
pub struct SimConfig {
    pub width: u16,
    pub height: u16,
    /// layer count IS a config array — Spike 4 decides it
    pub slots: Vec<LayerSlot>,
    pub seed: u64,
    /// `None` -> the embedded table
    pub material_path: Option<std::path::PathBuf>,
    /// true: bisect at startup; false: analytic seed
    pub calibrate_repose: bool,
    pub parallel: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 512,
            slots: vec![
                LayerSlot::Face,
                LayerSlot::Plant,
                LayerSlot::Cavity,
                LayerSlot::Gangway,
            ],
            seed: 0x5EED_1234_ABCD_0001,
            material_path: None,
            calibrate_repose: true,
            parallel: true,
        }
    }
}

impl Materials {
    pub fn id(&self, name: &str) -> Option<crate::material::MaterialId> {
        self.0.id(name)
    }
}
