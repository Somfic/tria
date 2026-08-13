//! Coarse 8x8 grid hooks. Only `flow_mag` and `activity` are written in the slice.

use crate::units::CELL_PX;

pub struct CoarseGrid {
    /// `= w / CELL_PX`, `h / CELL_PX`
    pub cw: u16,
    pub ch: u16,
    /// allocated, never stepped
    pub heat: Vec<f32>,
    /// allocated, never stepped
    pub pressure: Vec<f32>,
    /// WRITTEN: liquid moves per coarse cell, decayed — feeds re-suspension
    pub flow_mag: Vec<u8>,
    /// WRITTEN: moves per coarse cell, debug overlay
    pub activity: Vec<u8>,
    /// present and unused, per the layer contract table
    pub cross_layer_coeff: f32,
}

impl CoarseGrid {
    pub fn new(w: u16, h: u16) -> Self {
        let cw = w.div_ceil(CELL_PX);
        let ch = h.div_ceil(CELL_PX);
        let n = cw as usize * ch as usize;
        Self {
            cw,
            ch,
            heat: vec![0.0; n],
            pressure: vec![0.0; n],
            flow_mag: vec![0; n],
            activity: vec![0; n],
            cross_layer_coeff: 0.0,
        }
    }

    #[inline]
    pub fn cell_of(&self, x: u16, y: u16) -> usize {
        (y / CELL_PX) as usize * self.cw as usize + (x / CELL_PX) as usize
    }

    #[inline]
    pub fn note_flow(&mut self, x: u16, y: u16) {
        let c = self.cell_of(x, y);
        if let Some(v) = self.flow_mag.get_mut(c) {
            *v = v.saturating_add(8);
        }
    }

    /// any move, powder or liquid — the debug activity overlay
    #[inline]
    pub fn note_activity(&mut self, x: u16, y: u16) {
        let c = self.cell_of(x, y);
        if let Some(v) = self.activity.get_mut(c) {
            *v = v.saturating_add(1);
        }
    }

    #[inline]
    pub fn flow_at(&self, x: u16, y: u16) -> u8 {
        self.flow_mag[self.cell_of(x, y)]
    }

    /// called at `COARSE_EVERY`: `flow_mag -= flow_mag / 4`, `activity = 0`
    pub fn decay(&mut self) {
        for v in &mut self.flow_mag {
            *v -= *v / 4;
        }
        self.activity.fill(0);
    }
}

/// Layer-agnostic gas plane — gases ignore layers, so the topology is baked in now.
pub struct GasPlane {
    pub cw: u16,
    pub ch: u16,
    pub concentration: Vec<f32>,
    pub temperature: Vec<f32>,
}

impl GasPlane {
    pub fn new(w: u16, h: u16) -> Self {
        let cw = w.div_ceil(CELL_PX);
        let ch = h.div_ceil(CELL_PX);
        let n = cw as usize * ch as usize;
        Self {
            cw,
            ch,
            concentration: vec![0.0; n],
            // ambient, degrees C
            temperature: vec![20.0; n],
        }
    }
}
