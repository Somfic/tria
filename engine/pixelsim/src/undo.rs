//! Sim-aware undo. A cell is restored only if it is still the player's — the
//! `FLAG_PLAYER_PLACED` bit travels with a cell when the solver moves it, so a cell
//! the sim has since overwritten no longer carries the bit and sim history is never
//! resurrected.

use crate::cell::{CellRect, FLAG_PLAYER_PLACED};
use crate::layer::Layer;

/// bytes charged per RLE run: 9 channels + the u16 run length
const RUN_BYTES: usize = 11;

pub struct Stroke {
    pub layer: u16,
    pub rect: CellRect,
    /// RLE: `(run_len, prior channels)`
    pub runs: Vec<(u16, [u8; 9])>,
}

pub struct UndoJournal {
    strokes: std::collections::VecDeque<Stroke>,
    bytes: usize,
    max_strokes: usize,
    max_bytes: usize,
}

impl UndoJournal {
    /// 64 strokes / 32 MB
    pub fn new() -> Self {
        Self {
            strokes: std::collections::VecDeque::new(),
            bytes: 0,
            max_strokes: 64,
            max_bytes: 32 * 1024 * 1024,
        }
    }

    /// Snapshots the prior state of `rect`, RLE-compressed in row-major order. Runs of
    /// identical cells collapse, so a 128x128 erase of open air costs one run.
    pub fn begin(&mut self, layer: &Layer, layer_index: u16, rect: CellRect) {
        let r = rect.clamped(layer.w, layer.h);
        if r.width() == 0 || r.height() == 0 {
            return;
        }
        let mut runs: Vec<(u16, [u8; 9])> = Vec::new();
        for y in r.y0..=r.y1 {
            for x in r.x0..=r.x1 {
                let c = layer.read_cell(layer.idx(x, y));
                match runs.last_mut() {
                    Some((n, prev)) if *prev == c && *n < u16::MAX => *n += 1,
                    _ => runs.push((1, c)),
                }
            }
        }
        self.bytes += runs.len() * RUN_BYTES;
        self.strokes.push_back(Stroke {
            layer: layer_index,
            rect: r,
            runs,
        });
    }

    /// Closes the pending stroke and enforces the depth/size budget.
    pub fn commit(&mut self) {
        while self.strokes.len() > self.max_strokes
            || (self.bytes > self.max_bytes && self.strokes.len() > 1)
        {
            let Some(s) = self.strokes.pop_front() else {
                break;
            };
            self.bytes = self.bytes.saturating_sub(s.runs.len() * RUN_BYTES);
        }
    }

    pub fn undo(&mut self, layers: &mut [Layer]) -> bool {
        let Some(s) = self.strokes.pop_back() else {
            return false;
        };
        self.bytes = self.bytes.saturating_sub(s.runs.len() * RUN_BYTES);
        let li = s.layer as usize;
        if li >= layers.len() {
            return false;
        }
        let l = &mut layers[li];
        let width = s.rect.width() as usize;
        if width == 0 {
            return false;
        }

        let mut k = 0usize;
        for &(n, c) in &s.runs {
            for _ in 0..n {
                let x = s.rect.x0 + (k % width) as u16;
                let y = s.rect.y0 + (k / width) as u16;
                k += 1;
                if !l.in_bounds(x as i32, y as i32) {
                    continue;
                }
                let i = l.idx(x, y);
                if l.flags[i] & FLAG_PLAYER_PLACED == 0 {
                    continue;
                }
                l.write_cell(i, c);
            }
        }
        l.chunks.wake_rect(s.rect);
        l.chunks.mark_dirty_rect(s.rect);
        true
    }

    pub fn depth(&self) -> usize {
        self.strokes.len()
    }

    /// bytes currently held by the journal
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Default for UndoJournal {
    fn default() -> Self {
        Self::new()
    }
}
