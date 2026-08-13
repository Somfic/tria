//! The two cues that make a stack of layers read as slabs in one space rather than
//! images on top of each other: a **lip** on every silhouette edge, and a **cast** from
//! each slab onto the slab behind it.
//!
//! Both are baked into the layer texture rather than rendered as geometry, for the reason
//! `depth.rs` gives: our planes are flat quads, so anything with real thickness has to be
//! drawn. That is not a compromise at our depths — an LBP slab face, seen through a narrow
//! lens a few slots from the camera, is a few px of darker lip on one side.
//!
//! Both passes are deliberately **camera-independent**. A lip that pointed at the vanishing
//! point, or a cast that moved with the camera, would invalidate every tile on every pan
//! and take the dirty-rect bake with it. So the light direction is a constant, and the
//! whole scene is lit from the upper left like every cutaway illustration ever drawn.

use pixelsim::{CHUNK_PX, Layer};

use crate::treat::{ExtrudeConfig, ShadowConfig};

/// Cells per mask cell. 4 is the coarsest that still resolves a chute wall.
pub const MASK_DIV: u16 = 4;

/// Coarse occupancy of one layer: `255` fully solid, `0` fully open.
///
/// This is what a slab casts *with*. Coarse because a shadow is low-frequency by nature —
/// a per-cell shadow mask would cost as much as the bake it feeds and look no different
/// once softened.
#[derive(Default, Clone)]
pub struct SlabMask {
    pub w: u16,
    pub h: u16,
    /// occupancy, `w * h`
    pub cells: Vec<u8>,
    /// scratch for the separable box blur
    blur: Vec<u8>,
}

impl SlabMask {
    pub fn for_layer(w: u16, h: u16) -> Self {
        let (mw, mh) = (w.div_ceil(MASK_DIV).max(1), h.div_ceil(MASK_DIV).max(1));
        Self {
            w: mw,
            h: mh,
            cells: vec![0; mw as usize * mh as usize],
            blur: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Recompute the mask cells covering an inclusive cell rect, then soften them.
    ///
    /// Softening is done over the *whole* mask rather than the rect, because it is 32k
    /// cells for a 1024x512 layer — 0.03 ms — and a rect-local blur would seam.
    pub fn update(&mut self, layer: &Layer, rects: &[(u16, u16, u16, u16)], all: bool) {
        if self.cells.is_empty() {
            return;
        }
        if all {
            self.recompute(layer, (0, 0, layer.w - 1, layer.h - 1));
        } else {
            for &r in rects {
                self.recompute(layer, r);
            }
        }
    }

    fn recompute(&mut self, layer: &Layer, rect: (u16, u16, u16, u16)) {
        let (x0, y0, x1, y1) = rect;
        let mx0 = x0 / MASK_DIV;
        let my0 = y0 / MASK_DIV;
        let mx1 = (x1 / MASK_DIV).min(self.w - 1);
        let my1 = (y1 / MASK_DIV).min(self.h - 1);
        for my in my0..=my1 {
            for mx in mx0..=mx1 {
                let mut solid = 0u32;
                let mut total = 0u32;
                for dy in 0..MASK_DIV {
                    let y = my * MASK_DIV + dy;
                    if y >= layer.h {
                        break;
                    }
                    let row = y as usize * layer.w as usize;
                    for dx in 0..MASK_DIV {
                        let x = mx * MASK_DIV + dx;
                        if x >= layer.w {
                            break;
                        }
                        total += 1;
                        if layer.mat[row + x as usize] != 0 {
                            solid += 1;
                        }
                    }
                }
                let v = if total == 0 {
                    0
                } else {
                    (solid * 255 / total) as u8
                };
                self.cells[my as usize * self.w as usize + mx as usize] = v;
            }
        }
    }

    /// Softened occupancy at a fractional mask coordinate, bilinear. Out of bounds reads
    /// as open, so a slab does not cast a shadow off the edge of the vessel.
    #[inline]
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        if self.cells.is_empty() {
            return 0.0;
        }
        let src = if self.blur.len() == self.cells.len() {
            &self.blur
        } else {
            &self.cells
        };
        let (w, h) = (self.w as i32, self.h as i32);
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let at = |ix: i32, iy: i32| -> f32 {
            if ix < 0 || iy < 0 || ix >= w || iy >= h {
                0.0
            } else {
                src[iy as usize * self.w as usize + ix as usize] as f32 * (1.0 / 255.0)
            }
        };
        let (ix, iy) = (x0 as i32, y0 as i32);
        let a = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
        let b = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
        a * (1.0 - fy) + b * fy
    }

    /// Box-blur the occupancy into `blur`, `radius` mask cells, separably.
    pub fn soften(&mut self, radius: u8) {
        if radius == 0 || self.cells.is_empty() {
            self.blur.clear();
            return;
        }
        let n = self.cells.len();
        if self.blur.len() != n {
            self.blur = vec![0; n];
        }
        let (w, h) = (self.w as usize, self.h as usize);
        let r = radius as usize;
        // horizontal into blur
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let lo = x.saturating_sub(r);
                let hi = (x + r).min(w - 1);
                let mut acc = 0u32;
                for sx in lo..=hi {
                    acc += self.cells[row + sx] as u32;
                }
                self.blur[row + x] = (acc / (hi - lo + 1) as u32) as u8;
            }
        }
        // vertical, in place over columns, reading the horizontal result
        for x in 0..w {
            let mut col = [0u8; 64];
            let use_scratch = h <= col.len();
            if use_scratch {
                for y in 0..h {
                    col[y] = self.blur[y * w + x];
                }
            }
            for y in 0..h {
                let lo = y.saturating_sub(r);
                let hi = (y + r).min(h - 1);
                let mut acc = 0u32;
                for sy in lo..=hi {
                    acc += if use_scratch {
                        col[sy] as u32
                    } else {
                        self.cells[sy * w + x] as u32
                    };
                }
                self.blur[y * w + x] = (acc / (hi - lo + 1) as u32) as u8;
            }
        }
    }
}

/// How far a cast is displaced, in cells, for a caster this many slots in front.
#[inline]
pub fn cast_offset(cfg: &ShadowConfig, dir: (f32, f32), slots: f32) -> (f32, f32) {
    let d = cfg.offset_px * slots.max(0.0);
    (dir.0 * d, dir.1 * d)
}

/// Where a chunk index lands, as an inclusive cell rect. Mirrors `bake::chunk_rect` for
/// callers that only have the sim's dirty list.
#[inline]
pub fn chunk_cell_rect(chunk: u32, cw: usize, w: u16, h: u16) -> Option<(u16, u16, u16, u16)> {
    let cx = (chunk as usize % cw) as u16;
    let cy = (chunk as usize / cw) as u16;
    let x0 = cx * CHUNK_PX;
    let y0 = cy * CHUNK_PX;
    if x0 >= w || y0 >= h {
        return None;
    }
    Some((x0, y0, (x0 + CHUNK_PX - 1).min(w - 1), (y0 + CHUNK_PX - 1).min(h - 1)))
}

/// The lip pass, per pixel: is this air cell inside the lip thrown by a solid cell, and
/// if so how deep into it?
///
/// Returns the source cell to take the colour from and how far to shade it, or `None`
/// when the pixel is not part of a lip. Walking *back* along the light direction from an
/// air cell — rather than forward from every solid cell — is what keeps this a read-only
/// gather that a tile bake can run without touching its neighbours' pixels.
#[inline]
pub fn lip_at(layer: &Layer, cfg: &ExtrudeConfig, x: u16, y: u16, depth: f32) -> Option<(u16, u16, f32)> {
    let n = depth.round() as i32;
    if n <= 0 {
        return None;
    }
    for k in 1..=n {
        let sx = x as f32 - cfg.dir.0 * k as f32;
        let sy = y as f32 - cfg.dir.1 * k as f32;
        let (sxi, syi) = (sx.round() as i32, sy.round() as i32);
        if !layer.in_bounds(sxi, syi) {
            return None;
        }
        let (sxu, syu) = (sxi as u16, syi as u16);
        if layer.mat[layer.idx(sxu, syu)] != 0 {
            // deeper into the lip is darker, so the face reads as curving away
            let t = k as f32 / n as f32;
            return Some((sxu, syu, cfg.shade * (0.55 + 0.45 * t)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelsim::{LayerSlot, MaterialTable};

    fn layer_with_block() -> (Layer, u8) {
        let t = MaterialTable::embedded();
        let stone = t.id("stone").unwrap().0;
        let mut l = Layer::new(64, 64, LayerSlot::Plant, 7);
        for y in 20..30u16 {
            for x in 20..30u16 {
                let i = l.idx(x, y);
                l.mat[i] = stone;
            }
        }
        (l, stone)
    }

    #[test]
    fn the_mask_measures_coverage_not_presence() {
        let (l, _) = layer_with_block();
        let mut m = SlabMask::for_layer(l.w, l.h);
        m.update(&l, &[], true);
        // interior of the block: fully solid
        assert_eq!(m.sample(6.0, 6.0), 1.0);
        // well outside: open
        assert_eq!(m.sample(1.0, 1.0), 0.0);
        // a mask cell straddling the block edge is partial, not 0 or 1
        let edge = m.cells[(5 * m.w + 7) as usize];
        assert!(edge > 0 && edge < 255, "straddling cell was {edge}");
    }

    /// Softening must spread occupancy outward — that is the whole point — but must not
    /// invent occupancy where there is nothing anywhere near.
    #[test]
    fn softening_spreads_but_does_not_invent() {
        let (l, _) = layer_with_block();
        let mut m = SlabMask::for_layer(l.w, l.h);
        m.update(&l, &[], true);
        let before = m.sample(4.0, 6.0);
        m.soften(2);
        let after = m.sample(4.0, 6.0);
        assert!(after > before, "shadow must spread past the caster's edge");
        assert_eq!(m.sample(0.0, 0.0), 0.0, "far from any caster stays lit");
    }

    #[test]
    fn a_lip_is_thrown_on_one_side_only() {
        let (l, _) = layer_with_block();
        let cfg = ExtrudeConfig::default();
        // down-right of the block: inside the lip
        assert!(lip_at(&l, &cfg, 30, 31, 3.0).is_some());
        // up-left of the block: no lip, the light comes from there
        assert!(lip_at(&l, &cfg, 19, 18, 3.0).is_none());
        // far away: nothing
        assert!(lip_at(&l, &cfg, 50, 50, 3.0).is_none());
    }

    #[test]
    fn the_lip_darkens_with_distance_from_the_face() {
        let (l, _) = layer_with_block();
        let cfg = ExtrudeConfig::default();
        // straight below the block's bottom edge, one and two cells out. Diagonally the
        // lip is thinner than `depth` suggests, because the direction is not axis-aligned.
        let near = lip_at(&l, &cfg, 25, 30, 3.0).map(|(_, _, s)| s).unwrap();
        let far = lip_at(&l, &cfg, 25, 31, 3.0).map(|(_, _, s)| s).unwrap();
        assert!(far > near, "deeper into the lip must be darker: {near} vs {far}");
    }

    /// A cast is displaced further for a caster further in front, or the stack has no
    /// sense of how deep the gap between two slabs is.
    #[test]
    fn cast_offset_grows_with_the_gap() {
        let cfg = ShadowConfig::default();
        let dir = ExtrudeConfig::default().dir;
        let (x1, y1) = cast_offset(&cfg, dir, 1.0);
        let (x2, y2) = cast_offset(&cfg, dir, 2.0);
        assert!(x2 > x1 && y2 > y1);
        assert_eq!(cast_offset(&cfg, dir, 0.0), (0.0, 0.0));
    }
}
