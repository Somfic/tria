//! Per-layer and aggregate sim measurements.

#[derive(Clone, Debug, Default)]
pub struct LayerStats {
    pub active_cells: u32,
    pub awake_chunks: u32,
    pub moves: u32,
    pub step_ms: f32,
    pub powder_ms: f32,
    pub liquid_ms: f32,
    pub wet_ms: f32,
    pub slurry_ms: f32,
    /// `liquid::compute_head` — two more full sweeps of the awake area
    pub head_ms: f32,
    /// `clear_moved_flags` + `mark_slept` + the chunk-map bookkeeping
    pub book_ms: f32,
    /// dust generated this tick: `base * (1.0 - wetness)`
    pub dust: f32,
}

impl LayerStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Debug, Default)]
pub struct SimStats {
    pub active_cells: u32,
    pub awake_chunks: u32,
    pub moves: u32,
    pub step_ms: f32,
    pub powder_ms: f32,
    pub liquid_ms: f32,
    pub wet_ms: f32,
    pub slurry_ms: f32,
    pub head_ms: f32,
    pub book_ms: f32,
    pub port_ms: f32,
    pub per_layer_ms: Vec<f32>,
    pub transfers: u32,
}

impl SimStats {
    pub fn reset(&mut self) {
        self.active_cells = 0;
        self.awake_chunks = 0;
        self.moves = 0;
        self.step_ms = 0.0;
        self.powder_ms = 0.0;
        self.liquid_ms = 0.0;
        self.wet_ms = 0.0;
        self.slurry_ms = 0.0;
        self.head_ms = 0.0;
        self.book_ms = 0.0;
        self.port_ms = 0.0;
        self.transfers = 0;
        // keep the allocation: this resource is rebuilt every tick
        self.per_layer_ms.clear();
    }

    /// Sums a layer's counters and pushes its wall time onto `per_layer_ms`. The
    /// phase millisecond figures are summed across layers, so with `parallel` on they
    /// exceed `step_ms` — that is the point, it shows the parallel win.
    pub fn accumulate(&mut self, layer: &LayerStats) {
        self.active_cells += layer.active_cells;
        self.awake_chunks += layer.awake_chunks;
        self.moves += layer.moves;
        self.powder_ms += layer.powder_ms;
        self.liquid_ms += layer.liquid_ms;
        self.wet_ms += layer.wet_ms;
        self.slurry_ms += layer.slurry_ms;
        self.head_ms += layer.head_ms;
        self.book_ms += layer.book_ms;
        self.per_layer_ms.push(layer.step_ms);
    }
}
