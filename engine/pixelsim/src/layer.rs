//! One independent solver domain. Solids never cross layers, so every solver
//! function takes exactly one `&mut Layer`.

use crate::cell::{CellRect, FLAG_MOVED, FLAG_SLEEPING};
use crate::chunk::ChunkMap;
use crate::coarse::CoarseGrid;
use crate::rng::Rng;
use crate::stats::LayerStats;
use crate::units::LayerSlot;

/// Channel order of the packed 9-byte cell used by undo strokes and the clipboard.
pub const CH_MAT: usize = 0;
pub const CH_FLAGS: usize = 1;
pub const CH_WETNESS: usize = 2;
pub const CH_CHARRED: usize = 3;
pub const CH_DIRT: usize = 4;
pub const CH_WEAR: usize = 5;
pub const CH_HEAD: usize = 6;
pub const CH_SUSP_MAT: usize = 7;
pub const CH_SUSP_CONC: usize = 8;

pub struct Layer {
    pub w: u16,
    pub h: u16,
    pub slot: LayerSlot,
    pub depth_m: f32,
    pub standable: bool,
    pub crouch_only: bool,
    /// +y is down; `[0.0, 1.0]` in the slice. no `Vec2` — this module is bevy-free
    pub gravity: [f32; 2],
    /// authoritative material id
    pub mat: Vec<u8>,
    /// `FLAG_*` bits, see `cell.rs`
    pub flags: Vec<u8>,
    pub wetness: Vec<u8>,
    /// allocated, never written in the slice
    pub charred: Vec<u8>,
    pub dirt: Vec<u8>,
    pub wear: Vec<u8>,
    pub head: Vec<u8>,
    pub susp_mat: Vec<u8>,
    pub susp_conc: Vec<u8>,
    pub chunks: ChunkMap,
    pub coarse: CoarseGrid,
    /// per-layer stream — threading cannot perturb determinism
    pub rng: Rng,
    pub tick: u64,
    pub stats: LayerStats,
}

impl Layer {
    #[inline]
    pub fn idx(&self, x: u16, y: u16) -> usize {
        y as usize * self.w as usize + x as usize
    }

    #[inline]
    pub fn in_bounds(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.w as u32 && (y as u32) < self.h as u32
    }

    #[inline]
    pub fn mat_at(&self, x: u16, y: u16) -> u8 {
        self.mat[self.idx(x, y)]
    }

    /// marks the chunk dirty + wakes neighbours
    #[inline]
    pub fn set_mat(&mut self, x: u16, y: u16, m: u8) {
        let i = self.idx(x, y);
        self.mat[i] = m;
        self.flags[i] &= !FLAG_SLEEPING;
        self.chunks.mark_dirty(x, y);
        self.chunks.wake(x, y);
        // painted water has to soak in even if it never moves — see `touch`
        self.chunks.mark_damp(x, y);
    }

    pub fn new(w: u16, h: u16, slot: LayerSlot, seed: u64) -> Self {
        let n = w as usize * h as usize;
        Self {
            w,
            h,
            slot,
            depth_m: slot.depth_m(),
            standable: slot.standable(),
            crouch_only: slot.crouch_only(),
            gravity: [0.0, 1.0],
            mat: vec![0; n],
            flags: vec![0; n],
            wetness: vec![0; n],
            charred: vec![0; n],
            dirt: vec![0; n],
            wear: vec![0; n],
            head: vec![0; n],
            susp_mat: vec![0; n],
            susp_conc: vec![0; n],
            chunks: ChunkMap::new(w, h),
            coarse: CoarseGrid::new(w, h),
            rng: Rng::new(seed),
            tick: 0,
            stats: LayerStats::default(),
        }
    }

    /// Fills every cell with `m`, resets the aux channels, and wakes everything.
    pub fn fill(&mut self, m: u8) {
        self.mat.fill(m);
        self.flags.fill(0);
        self.wetness.fill(0);
        self.charred.fill(0);
        self.dirt.fill(0);
        self.wear.fill(0);
        self.head.fill(0);
        self.susp_mat.fill(0);
        self.susp_conc.fill(0);
        let n = self.chunks.len();
        self.chunks.active.set_range(n);
        self.chunks.next.set_range(n);
        self.chunks.dirty.set_range(n);
        self.chunks.damp.set_range(n);
        self.chunks.still.fill(0);
    }

    /// Clears `FLAG_MOVED` and `FLAG_SLEEPING` across awake chunks only — the whole
    /// point of the chunk map is that a settled layer costs nothing per tick.
    pub fn clear_moved_flags(&mut self) {
        const KEEP: u8 = !(FLAG_MOVED | FLAG_SLEEPING);
        let (w, h) = (self.w, self.h);
        for c in 0..self.chunks.len() {
            if !self.chunks.active.get(c) {
                continue;
            }
            let r = self.chunks.bounds(c, w, h);
            for y in r.y0..=r.y1 {
                let row = y as usize * w as usize;
                for f in &mut self.flags[row + r.x0 as usize..=row + r.x1 as usize] {
                    *f &= KEEP;
                }
            }
        }
    }

    /// Paints `FLAG_SLEEPING` on chunks that just went quiet, so the renderer can
    /// tint dead zones. Called once per tick after `ChunkMap::end_tick`.
    pub fn mark_slept(&mut self) {
        let (w, h) = (self.w, self.h);
        for c in 0..self.chunks.len() {
            if !(self.chunks.active.get(c) && !self.chunks.next.get(c)) {
                continue;
            }
            let r = self.chunks.bounds(c, w, h);
            for y in r.y0..=r.y1 {
                let row = y as usize * w as usize;
                for f in &mut self.flags[row + r.x0 as usize..=row + r.x1 as usize] {
                    *f |= FLAG_SLEEPING;
                }
            }
            self.chunks.dirty.set(c);
        }
    }

    // ---- cell primitives shared by every solver ------------------------------

    /// Swaps all nine channels between two cells. Aux state always travels with the
    /// material — this is the only way a cell ever moves.
    #[inline]
    pub(crate) fn swap_cells(&mut self, a: usize, b: usize) {
        self.mat.swap(a, b);
        self.flags.swap(a, b);
        self.wetness.swap(a, b);
        self.charred.swap(a, b);
        self.dirt.swap(a, b);
        self.wear.swap(a, b);
        self.head.swap(a, b);
        self.susp_mat.swap(a, b);
        self.susp_conc.swap(a, b);
    }

    /// Moves the cell at `from` to `to` (a swap, so the displaced content ends up at
    /// `from`), then does all the bookkeeping a move owes: moved flag, chunk wake,
    /// dirty marks, coarse activity, move count.
    #[inline]
    pub(crate) fn commit_move(&mut self, from: (u16, u16), to: (u16, u16)) {
        let a = self.idx(from.0, from.1);
        let b = self.idx(to.0, to.1);
        self.swap_cells(a, b);
        // BOTH ends are retired for this tick. A move is a swap, so whatever occupied
        // `to` is now sitting at `from` — displaced, i.e. it moved. Marking only `to`
        // let the displaced content keep the clean flag byte the swap handed it and
        // move again later in the same scan: a water cell shoved aside by sinking grit
        // then raced several more cells before the tick ended.
        self.flags[a] |= FLAG_MOVED;
        self.flags[b] |= FLAG_MOVED;
        self.flags[b] &= !FLAG_SLEEPING;
        self.flags[a] &= !FLAG_SLEEPING;
        self.chunks.wake(from.0, from.1);
        self.chunks.wake(to.0, to.1);
        self.chunks.mark_dirty(from.0, from.1);
        self.chunks.mark_dirty(to.0, to.1);
        self.coarse.note_activity(to.0, to.1);
        self.stats.moves += 1;
    }

    /// Marks a cell changed in place (aux channel edit, deposit, entrainment) without
    /// counting it as a move.
    #[inline]
    pub(crate) fn touch(&mut self, x: u16, y: u16) {
        self.chunks.mark_dirty(x, y);
        self.chunks.wake(x, y);
        // A changed cell may be liquid that has just come to rest against absorbent
        // powder. The wet passes run off the damp set precisely so they survive sleep, so
        // a cell that changes and then never moves again must leave a mark they will find.
        // One bit, and the passes prune it as soon as the chunk turns out to be dry.
        self.chunks.mark_damp(x, y);
    }

    /// Marks a cell changed for the *renderer* only, without waking the solvers.
    ///
    /// Used by the aux rules: a wet cell losing one unit of wetness has to be re-baked,
    /// but waking its chunk would put it back in the awake set every aux tick and
    /// dissolve the sleep win entirely.
    #[inline]
    pub(crate) fn touch_render(&mut self, x: u16, y: u16) {
        self.chunks.mark_dirty(x, y);
    }

    /// Records that `(x, y)` may now hold wetness, so the low-rate aux pass visits it
    /// even after its chunk has gone to sleep.
    #[inline]
    pub(crate) fn mark_damp(&mut self, x: u16, y: u16) {
        self.chunks.mark_damp(x, y);
    }

    #[inline]
    pub fn read_cell(&self, i: usize) -> [u8; 9] {
        [
            self.mat[i],
            self.flags[i],
            self.wetness[i],
            self.charred[i],
            self.dirt[i],
            self.wear[i],
            self.head[i],
            self.susp_mat[i],
            self.susp_conc[i],
        ]
    }

    #[inline]
    pub fn write_cell(&mut self, i: usize, c: [u8; 9]) {
        self.mat[i] = c[CH_MAT];
        self.flags[i] = c[CH_FLAGS];
        self.wetness[i] = c[CH_WETNESS];
        self.charred[i] = c[CH_CHARRED];
        self.dirt[i] = c[CH_DIRT];
        self.wear[i] = c[CH_WEAR];
        self.head[i] = c[CH_HEAD];
        self.susp_mat[i] = c[CH_SUSP_MAT];
        self.susp_conc[i] = c[CH_SUSP_CONC];
        // undo strokes, the clipboard and port transfers can all paste wetness into a
        // sleeping chunk; the aux pass has to hear about it
        if c[CH_WETNESS] > 0 && self.w > 0 {
            let (x, y) = ((i % self.w as usize) as u16, (i / self.w as usize) as u16);
            self.chunks.mark_damp(x, y);
        }
    }

    /// Zeroes every aux channel of a cell, leaving `mat` and `flags` alone.
    #[inline]
    pub(crate) fn clear_aux(&mut self, i: usize) {
        self.wetness[i] = 0;
        self.charred[i] = 0;
        self.dirt[i] = 0;
        self.wear[i] = 0;
        self.head[i] = 0;
        self.susp_mat[i] = 0;
        self.susp_conc[i] = 0;
    }

    /// The whole layer as a rect.
    pub fn rect(&self) -> CellRect {
        CellRect {
            x0: 0,
            y0: 0,
            x1: self.w.saturating_sub(1),
            y1: self.h.saturating_sub(1),
        }
    }

    /// FNV-1a over `mat` — the determinism test's grid hash.
    pub fn hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in &self.mat {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for &b in &self.wetness {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        for &b in &self.susp_conc {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }
}
