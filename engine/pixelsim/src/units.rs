//! Every magic number in the sim, with its unit.

/// metres per sim pixel
pub const PX_M: f32 = 0.10;
/// coarse sim cell edge, in sim pixels — also the machine grid cell
pub const CELL_PX: u16 = 8;
/// sleep granularity, in sim pixels — 8x8 coarse cells
pub const CHUNK_PX: u16 = 64;
/// player height in sim pixels
pub const PLAYER_PX: u16 = 16;
/// fixed sim tick rate, Hz
pub const TICK_HZ: f64 = 60.0;
/// coarse grid update cadence, in ticks (15 Hz)
pub const COARSE_EVERY: u64 = 4;
/// liquid body/head pass cadence, in ticks (5 Hz)
pub const HEAD_EVERY: u64 = 12;
/// depth of a thick (standable) layer, metres
pub const DEPTH_THICK_M: f32 = 0.8;
/// depth of a thin (crouch-only / cavity) layer, metres
pub const DEPTH_THIN_M: f32 = 0.2;
/// consecutive zero-move ticks before a chunk sleeps
pub const SLEEP_TICKS: u8 = 8;
/// cadence of the low-rate auxiliary pass, in ticks.
///
/// Chunk sleep is a *motion* predicate: a settled pile stops moving after
/// `SLEEP_TICKS` and then never runs a solver pass again. The slow aux rules
/// (evaporation at `EVAPORATE_EVERY`, drip at `DRIP_EVERY`) are on periods far longer
/// than that, so hanging them off the awake set freezes them permanently. They get
/// their own low-rate schedule instead, driven by the damp-chunk set, which is
/// independent of sleep. Must divide `EVAPORATE_EVERY` and `DRIP_EVERY`.
pub const AUX_EVERY: u64 = 8;
/// grains dropped per bisection trial by the startup repose calibration. 4000 would
/// be more faithful, but costs seconds of startup for a sub-degree gain.
pub const CALIBRATE_GRAINS: u32 = 1200;

/// Which slot in the front-to-back layer stack a layer occupies.
///
/// Four slots, front to back, per the vault's `One Simulated Plane` decision. `Deck`,
/// `Works` and `Hold` used to be three of six *slots*; they are now vertical **floors**
/// inside [`LayerSlot::Plant`], so the factory grows tall rather than deep.
///
/// The load-bearing consequence: **only `Plant` is simulated.** The other three slabs are
/// static structure — geometry the player builds, walks on and routes through, but which no
/// solver ever visits. See [`LayerSlot::simulated`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum LayerSlot {
    /// signage, decoration, external fittings
    Face,
    /// the simulated plane: all material, machines, flow, fire, flood
    Plant,
    /// services — wiring, pipes, ducts, all abstract transport
    Cavity,
    /// circulation, stairs, crew, crated storage, access
    Gangway,
}

impl LayerSlot {
    /// Front to back.
    pub const ALL: [LayerSlot; 4] = [
        LayerSlot::Face,
        LayerSlot::Plant,
        LayerSlot::Cavity,
        LayerSlot::Gangway,
    ];

    /// Does the falling-sand solver run in this slab?
    ///
    /// Exactly one slot says yes. This is the whole of `One Simulated Plane` expressed as a
    /// predicate, and everything downstream — the step loop, the bake rates, whether a
    /// slab's geometry can be meshed once and forgotten — keys off it.
    pub fn simulated(self) -> bool {
        matches!(self, LayerSlot::Plant)
    }

    /// thick slabs are standable; thin ones are decoration or service runs
    pub fn thick(self) -> bool {
        matches!(self, LayerSlot::Plant | LayerSlot::Gangway)
    }

    /// nominal depth of the slot in metres
    pub fn depth_m(self) -> f32 {
        if self.thick() {
            DEPTH_THICK_M
        } else {
            DEPTH_THIN_M
        }
    }

    /// can the player walk here
    pub fn standable(self) -> bool {
        self.thick()
    }

    /// can the player only crouch here
    pub fn crouch_only(self) -> bool {
        matches!(self, LayerSlot::Cavity)
    }
}
