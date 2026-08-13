//! Vessel-at-origin indirection and world <-> cell math.

/// The sim always runs with the vessel at the origin; this frame is the only place
/// that knows where the vessel actually is in world space.
///
/// Both functions work in *sim space*: one world unit is one sim pixel, `+x` right and
/// `+y` down, with `origin` the position of cell `(0, 0)`'s top-left corner. Turning a
/// y-up engine position into sim space is the renderer's job, not the sim's.
#[derive(Copy, Clone, Debug, Default)]
pub struct VesselFrame {
    /// vessel origin in world units (sim px)
    pub origin: [f32; 2],
    /// vessel size in cells
    pub w: u16,
    pub h: u16,
}

impl VesselFrame {
    pub fn new(w: u16, h: u16) -> Self {
        Self {
            origin: [0.0, 0.0],
            w,
            h,
        }
    }

    /// world position -> cell, `None` when outside the vessel
    #[inline]
    pub fn world_to_cell(&self, world: [f32; 2]) -> Option<(u16, u16)> {
        let lx = world[0] - self.origin[0];
        let ly = world[1] - self.origin[1];
        if !(lx >= 0.0) || !(ly >= 0.0) {
            return None;
        }
        let cx = lx.floor() as i64;
        let cy = ly.floor() as i64;
        if cx >= self.w as i64 || cy >= self.h as i64 {
            return None;
        }
        Some((cx as u16, cy as u16))
    }

    /// cell centre -> world position
    #[inline]
    pub fn cell_to_world(&self, cell: (u16, u16)) -> [f32; 2] {
        [
            self.origin[0] + cell.0 as f32 + 0.5,
            self.origin[1] + cell.1 as f32 + 0.5,
        ]
    }
}
