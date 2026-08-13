//! Chunk-granular sleep and dirty-rect bookkeeping.

use crate::cell::{Bitset, CellRect};
use crate::units::{CHUNK_PX, SLEEP_TICKS};

pub struct ChunkMap {
    /// chunk counts
    pub cw: u16,
    pub ch: u16,
    /// chunks to step this tick
    pub active: Bitset,
    /// accumulated for next tick
    pub next: Bitset,
    /// consumed by the renderer, cleared by `take_dirty`
    pub dirty: Bitset,
    /// Chunks that may hold wetness **or liquid**, and therefore owe the wet passes a
    /// visit.
    ///
    /// Deliberately *not* a subset of `active`: sleep means "nothing is moving here",
    /// which is exactly the state in which a wet pile still has to dry and a still pool
    /// still has to soak into the sand under it. Set conservatively (any wetness write,
    /// any cell change, any external edit) and pruned by the passes themselves, which
    /// clear the bit of every chunk they scan and find neither wet nor wet-able — so a
    /// vessel with no water in it converges to zero wet cost.
    pub damp: Bitset,
    /// consecutive zero-move ticks per chunk
    pub still: Vec<u8>,
    /// consecutive wet passes in which this chunk moved no water — see [`ChunkMap::wet_idle`]
    pub wet_still: Vec<u8>,
}

/// wet passes with no water moved before a chunk drops to the idle cadence
pub const WET_FUSE: u8 = 8;
/// ticks between visits to an idle damp chunk
pub const WET_IDLE_EVERY: u64 = 8;

impl ChunkMap {
    pub fn new(w: u16, h: u16) -> Self {
        let cw = w.div_ceil(CHUNK_PX);
        let ch = h.div_ceil(CHUNK_PX);
        let n = cw as usize * ch as usize;
        let mut active = Bitset::new(n);
        let mut next = Bitset::new(n);
        let mut dirty = Bitset::new(n);
        let mut damp = Bitset::new(n);
        // a fresh layer is fully awake and fully dirty, so whatever the caller fills
        // in settles from tick 0 and the renderer's first bake is a full bake.
        // `damp` starts set for the same reason — a caller that writes `wetness`
        // straight into the arrays has not told us where; the first aux pass prunes it
        // back to the chunks that really hold water.
        active.set_range(n);
        next.set_range(n);
        dirty.set_range(n);
        damp.set_range(n);
        Self {
            cw,
            ch,
            active,
            next,
            dirty,
            damp,
            still: vec![0; n],
            wet_still: vec![0; n],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.cw as usize * self.ch as usize
    }

    #[inline]
    pub fn chunk_of(&self, x: u16, y: u16) -> usize {
        (y / CHUNK_PX) as usize * self.cw as usize + (x / CHUNK_PX) as usize
    }

    /// wakes the containing chunk plus the 8 neighbours when on a boundary cell
    #[inline]
    pub fn wake(&mut self, x: u16, y: u16) {
        let cx = (x / CHUNK_PX) as i32;
        let cy = (y / CHUNK_PX) as i32;
        let c = cy as usize * self.cw as usize + cx as usize;
        self.next.set(c);
        if let Some(s) = self.still.get_mut(c) {
            *s = 0;
        }

        let lx = x % CHUNK_PX;
        let ly = y % CHUNK_PX;
        if !(lx == 0 || lx == CHUNK_PX - 1 || ly == 0 || ly == CHUNK_PX - 1) {
            return;
        }
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx >= self.cw as i32 || ny >= self.ch as i32 {
                    continue;
                }
                let nc = ny as usize * self.cw as usize + nx as usize;
                self.next.set(nc);
                self.still[nc] = 0;
            }
        }
    }

    /// wakes every chunk the rect touches plus a one-chunk border, this tick and next
    pub fn wake_rect(&mut self, r: CellRect) {
        if r.width() == 0 || r.height() == 0 || self.cw == 0 || self.ch == 0 {
            return;
        }
        let cx0 = (r.x0 / CHUNK_PX) as i32 - 1;
        let cx1 = (r.x1 / CHUNK_PX) as i32 + 1;
        let cy0 = (r.y0 / CHUNK_PX) as i32 - 1;
        let cy1 = (r.y1 / CHUNK_PX) as i32 + 1;
        for cy in cy0.max(0)..=cy1.min(self.ch as i32 - 1) {
            for cx in cx0.max(0)..=cx1.min(self.cw as i32 - 1) {
                let c = cy as usize * self.cw as usize + cx as usize;
                self.next.set(c);
                self.active.set(c);
                self.damp.set(c);
                self.still[c] = 0;
                self.wet_still[c] = 0;
            }
        }
    }

    /// Flags the chunk containing `(x, y)` as possibly holding wetness or liquid, and
    /// re-arms its wet pass. Cheap enough (one bit and one byte) to call on every wetness
    /// write and every cell change.
    #[inline]
    pub fn mark_damp(&mut self, x: u16, y: u16) {
        let c = self.chunk_of(x, y);
        self.damp.set(c);
        self.wet_still[c] = 0;
    }

    pub fn damp_count(&self) -> u32 {
        self.damp.count()
    }

    /// `true` when the wet pass may skip this chunk on this tick.
    ///
    /// A settled scene is the *common* case for the wet rules, not the rare one: a flooded
    /// vessel that has reached equilibrium still holds water against sand in every chunk,
    /// so the pass can neither sleep on motion nor prune the bit — it would have to keep
    /// re-scanning 400k cells a tick to discover, every tick, that there is nothing to do.
    /// Measured at 2.95 ms/tick on a flooded 1024x512 layer.
    ///
    /// So the pass gets a fuse of its own, exactly like [`SLEEP_TICKS`] for motion: a chunk
    /// where nothing has moved water for [`WET_FUSE`] passes drops to one visit per
    /// [`WET_IDLE_EVERY`] ticks. Unlike pruning the bit this cannot lose work — any change
    /// in the chunk re-arms it through `mark_damp`, and the worst case for anything else is
    /// that seepage starts up to eight ticks later, which is an eighth of a second.
    ///
    /// The phase is per chunk so idle chunks spread their visits across ticks instead of
    /// all landing on the same one.
    #[inline]
    pub fn wet_idle(&self, c: usize, tick: u64) -> bool {
        self.wet_still[c] >= WET_FUSE && (tick + c as u64) % WET_IDLE_EVERY != 0
    }

    /// Records what the wet pass found: `moved` resets the fuse, otherwise it burns down.
    #[inline]
    pub fn wet_worked(&mut self, c: usize, moved: bool) {
        if moved {
            self.wet_still[c] = 0;
        } else {
            self.wet_still[c] = self.wet_still[c].saturating_add(1);
        }
    }

    #[inline]
    pub fn mark_dirty(&mut self, x: u16, y: u16) {
        let c = self.chunk_of(x, y);
        self.dirty.set(c);
    }

    pub fn mark_dirty_rect(&mut self, r: CellRect) {
        if r.width() == 0 || r.height() == 0 || self.cw == 0 || self.ch == 0 {
            return;
        }
        for cy in (r.y0 / CHUNK_PX)..=(r.y1 / CHUNK_PX).min(self.ch - 1) {
            for cx in (r.x0 / CHUNK_PX)..=(r.x1 / CHUNK_PX).min(self.cw - 1) {
                self.dirty.set(cy as usize * self.cw as usize + cx as usize);
            }
        }
    }

    /// `active = next; next.clear_all()`
    pub fn begin_tick(&mut self) {
        core::mem::swap(&mut self.active, &mut self.next);
        self.next.clear_all();
    }

    /// resets the still counter, sets the next bit
    #[inline]
    pub fn note_move(&mut self, chunk: usize) {
        self.next.set(chunk);
        if let Some(s) = self.still.get_mut(chunk) {
            *s = 0;
        }
    }

    /// `still += 1` where no moves; sleeps at `SLEEP_TICKS`.
    ///
    /// A chunk that was awake but saw no moves is held awake for `SLEEP_TICKS` more
    /// ticks, so a pile that is still shuffling internally is never slept early.
    pub fn end_tick(&mut self) {
        for c in 0..self.still.len() {
            if self.next.get(c) {
                self.still[c] = 0;
            } else if self.active.get(c) {
                self.still[c] = self.still[c].saturating_add(1);
                if self.still[c] < SLEEP_TICKS {
                    self.next.set(c);
                }
            }
        }
    }

    /// chunks awake this tick that will not be awake next tick — the transition on
    /// which `FLAG_SLEEPING` gets painted
    pub fn newly_slept(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.still.len()).filter(|&c| self.active.get(c) && !self.next.get(c))
    }

    pub fn awake_count(&self) -> u32 {
        self.active.count()
    }

    /// the renderer's dirty-rect list; clears the bitset
    pub fn take_dirty(&mut self) -> Vec<usize> {
        let out: Vec<usize> = self.dirty.iter_set().collect();
        self.dirty.clear_all();
        out
    }

    /// Inclusive cell bounds of a chunk, clamped to the layer.
    ///
    /// The arithmetic is done in `u32`: the last chunk row of a layer taller than
    /// `65536 - CHUNK_PX` has `y0 + CHUNK_PX` past `u16::MAX`, and wrapping that made
    /// `min(h)` return 0 and `- 1` wrap back to 65535 — an out-of-range row index.
    #[inline]
    pub fn bounds(&self, chunk: usize, w: u16, h: u16) -> CellRect {
        let cx = (chunk % self.cw as usize) as u32;
        let cy = (chunk / self.cw as usize) as u32;
        let x0 = cx * CHUNK_PX as u32;
        let y0 = cy * CHUNK_PX as u32;
        CellRect {
            x0: x0 as u16,
            y0: y0 as u16,
            x1: ((x0 + CHUNK_PX as u32).min(w as u32) - 1) as u16,
            y1: ((y0 + CHUNK_PX as u32).min(h as u32) - 1) as u16,
        }
    }
}
