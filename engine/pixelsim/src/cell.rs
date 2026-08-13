//! Per-cell flag bits and the small geometry/bit helpers the solvers share.

pub const FLAG_MOVED: u8 = 1 << 0;
pub const FLAG_SLEEPING: u8 = 1 << 1;
pub const FLAG_PORT: u8 = 1 << 2;
pub const FLAG_PLAYER_PLACED: u8 = 1 << 3;
// bits 4..7 reserved

/// Inclusive cell rectangle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CellRect {
    pub x0: u16,
    pub y0: u16,
    pub x1: u16,
    pub y1: u16,
}

impl CellRect {
    /// Normalises the corners and clamps to a `w` x `h` grid. An empty grid yields
    /// the degenerate `0,0,0,0` rect.
    pub fn clamped(self, w: u16, h: u16) -> Self {
        if w == 0 || h == 0 {
            return Self {
                x0: 0,
                y0: 0,
                x1: 0,
                y1: 0,
            };
        }
        let (x0, x1) = if self.x0 <= self.x1 {
            (self.x0, self.x1)
        } else {
            (self.x1, self.x0)
        };
        let (y0, y1) = if self.y0 <= self.y1 {
            (self.y0, self.y1)
        } else {
            (self.y1, self.y0)
        };
        Self {
            x0: x0.min(w - 1),
            y0: y0.min(h - 1),
            x1: x1.min(w - 1),
            y1: y1.min(h - 1),
        }
    }

    pub fn width(self) -> u16 {
        if self.x1 >= self.x0 {
            self.x1 - self.x0 + 1
        } else {
            0
        }
    }

    pub fn height(self) -> u16 {
        if self.y1 >= self.y0 {
            self.y1 - self.y0 + 1
        } else {
            0
        }
    }

    pub fn contains(self, x: u16, y: u16) -> bool {
        x >= self.x0 && x <= self.x1 && y >= self.y0 && y <= self.y1
    }

    /// A single-cell rect.
    pub fn point(x: u16, y: u16) -> Self {
        Self {
            x0: x,
            y0: y,
            x1: x,
            y1: y,
        }
    }

    /// Smallest rect containing both — used to accumulate a stroke's touched area.
    pub fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }
}

pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    pub fn new(bits: usize) -> Self {
        Self {
            words: vec![0; bits.div_ceil(64)],
        }
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        let w = i >> 6;
        if w < self.words.len() {
            self.words[w] |= 1u64 << (i & 63);
        }
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        let w = i >> 6;
        if w < self.words.len() {
            self.words[w] &= !(1u64 << (i & 63));
        }
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        let w = i >> 6;
        w < self.words.len() && self.words[w] >> (i & 63) & 1 == 1
    }

    pub fn clear_all(&mut self) {
        self.words.fill(0);
    }

    /// Sets bits `0..n`. Never sets a bit past `n`, so `iter_set` and `count` cannot
    /// report an index outside the caller's logical range.
    pub fn set_range(&mut self, n: usize) {
        for i in 0..n {
            self.set(i);
        }
    }

    pub fn count(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Set bit indices, ascending — one `trailing_zeros` per set bit rather than one
    /// test per bit.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            core::iter::successors((word != 0).then_some(word), |&x| {
                let y = x & (x - 1);
                (y != 0).then_some(y)
            })
            .map(move |x| w * 64 + x.trailing_zeros() as usize)
        })
    }
}
