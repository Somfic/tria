//! Placeholder schematic bake: material-class blocks at 8x8 plus a per-layer tint.
//! Nothing beyond the placeholder is in scope for the slice.
//!
//! The vault is clear that this is eventually half the game ("machines as icons, power as
//! lines with thickness for load, flows as arrows") — so what exists here is deliberately
//! only the *seam*: one shared canvas at coarse resolution, composited back-to-front, so
//! the continuous-zoom crossfade in `zoom.rs` has something real to fade into and the
//! transition can be judged now rather than after symbols exist.
//!
//! # Why the cache is validated by content, not by dirty chunks
//!
//! Reducing a layer to class blocks means touching every cell, and while the crossfade is
//! up that happens every tick: three 1024x512 layers is 1.5 M cells of histogramming per
//! tick for a diagram that mostly does not change. The obvious fix — only rescan dirty
//! chunks — does not work here. `LayerCanvas::bake_with` drains `chunks.take_dirty()`
//! earlier in the same tick, and that list has exactly one reader by design, so by the time
//! the schematic runs it is empty. A second dirty list would make the sim owe the renderer
//! bookkeeping it has no business knowing about.
//!
//! So the cache validates itself: it keeps a 64-bit hash of each chunk's `mat` bytes
//! beside the coarse result and rescans a chunk only when its hash moves. Hashing is four
//! independent multiply chains over eight bytes at a time — several times cheaper per byte
//! than the class histogram it avoids — and it needs nothing from the sim and no
//! `&mut Layer`. It is also self-correcting about *identity*: a slot filled from some other
//! layer fails every hash comparison and is simply rebuilt.

use pixelsim::{CELL_PX, CellRect, Layer, MaterialClass, MaterialTable};

use crate::bake::LayerCanvas;

/// per-layer tint applied over the class blocks, indexed by depth from active
pub const SCHEMATIC_TINT: [[u8; 3]; 3] = [[90, 120, 150], [70, 95, 120], [50, 70, 90]];

/// class ink: powder ochre, liquid blue, solid slate, gas pale. Flat and diagrammatic on
/// purpose — a schematic that keeps the pixel palette is just a small pixel view.
pub const CLASS_INK: [[u8; 3]; 4] = [
    [198, 166, 100],
    [70, 130, 190],
    [140, 146, 152],
    [190, 200, 205],
];
/// how strongly the depth tint pulls the class ink
pub const TINT_MIX: f32 = 0.45;

/// a coarse texel with nothing in it
const EMPTY: u8 = 0xff;

/// How many composite passes one canvas caches. A vessel has a handful of layers; past
/// this, passes share the last slot, which stays *correct* — the hashes see to that — and
/// merely stops being a saving.
const MAX_PASSES: usize = 16;

/// Coarse results for one composite, keyed by position within it.
///
/// [`LayerCanvas::clear`] rewinds the pass counter, which is exactly what "callers must
/// `clear()` once and then walk layers back to front" already required.
#[derive(Default)]
pub struct SchematicCache {
    passes: Vec<Pass>,
    next: usize,
}

#[derive(Default)]
struct Pass {
    lw: u16,
    lh: u16,
    cw: u16,
    ch: u16,
    /// one hash per layer chunk
    hash: Vec<u64>,
    /// `(dominant class or EMPTY, coverage alpha)` per coarse texel
    cell: Vec<(u8, u8)>,
}

impl SchematicCache {
    /// Start a new composite.
    pub(crate) fn rewind(&mut self) {
        self.next = 0;
    }

    /// The slot for the next layer of this composite, reset if the layer or the canvas has
    /// changed shape under it.
    fn slot(&mut self, layer: &Layer, cw: u16, ch: u16) -> &mut Pass {
        let i = self.next.min(MAX_PASSES - 1);
        self.next += 1;
        if self.passes.len() <= i {
            self.passes.resize_with(i + 1, Pass::default);
        }
        let p = &mut self.passes[i];
        if p.lw != layer.w || p.lh != layer.h || p.cw != cw || p.ch != ch {
            p.lw = layer.w;
            p.lh = layer.h;
            p.cw = cw;
            p.ch = ch;
            p.hash.clear();
            p.hash.resize(layer.chunks.len(), 0);
            p.cell.clear();
            p.cell.resize(cw as usize * ch as usize, (EMPTY, 0));
        }
        if p.hash.len() != layer.chunks.len() {
            p.hash.clear();
            p.hash.resize(layer.chunks.len(), 0);
        }
        p
    }
}

/// Bake one layer into the shared schematic canvas at `CELL_PX` block resolution.
///
/// The canvas is one coarse texel per `CELL_PX` cell block; the sprite stretches it back
/// over the full vessel, so each texel reads as an 8x8 block. Composited with source-over
/// alpha, so callers must `clear()` once and then walk layers **back to front**.
///
/// Only chunks whose cells actually changed are rescanned; see the module docs for how that
/// is decided without a dirty list.
pub fn bake_schematic(canvas: &mut LayerCanvas, layer: &Layer, table: &MaterialTable, depth: i32) {
    let tint = SCHEMATIC_TINT[(depth.unsigned_abs() as usize).min(SCHEMATIC_TINT.len() - 1)];
    // the depth tint is per call, not per texel: four tinted inks instead of three lerps
    // per coarse block
    let ink: [[u8; 3]; 4] = core::array::from_fn(|k| {
        [
            mix(CLASS_INK[k][0], tint[0], TINT_MIX),
            mix(CLASS_INK[k][1], tint[1], TINT_MIX),
            mix(CLASS_INK[k][2], tint[2], TINT_MIX),
        ]
    });

    let (rgba, cw, ch, cache) = canvas.schematic_target();
    if cw == 0 || ch == 0 {
        return;
    }
    let pass = cache.slot(layer, cw, ch);

    // refresh the coarse result wherever the cells moved
    for c in 0..layer.chunks.len() {
        let r = layer.chunks.bounds(c, layer.w, layer.h);
        let h = hash_mat(layer, r);
        if pass.hash[c] == h {
            continue;
        }
        pass.hash[c] = h;
        rescan(pass, layer, table, r);
    }

    for (i, &(class, a)) in pass.cell.iter().enumerate() {
        if a == 0 || class == EMPTY {
            continue;
        }
        let o = i * 4;
        over(&mut rgba[o..o + 4], ink[class as usize], a);
    }
}

/// Re-derive the coarse texels covered by `r`.
fn rescan(pass: &mut Pass, layer: &Layer, table: &MaterialTable, r: CellRect) {
    let cw = pass.cw as usize;
    let ch = pass.ch as usize;
    if cw == 0 || ch == 0 {
        return;
    }
    for cy in (r.y0 / CELL_PX) as usize..=((r.y1 / CELL_PX) as usize).min(ch - 1) {
        for cx in (r.x0 / CELL_PX) as usize..=((r.x1 / CELL_PX) as usize).min(cw - 1) {
            // count the classes present in this CELL_PX x CELL_PX block
            let mut counts = [0u32; 4];
            let mut total = 0u32;
            let x0 = cx as u16 * CELL_PX;
            let y0 = cy as u16 * CELL_PX;
            for y in y0..(y0 + CELL_PX).min(layer.h) {
                for x in x0..(x0 + CELL_PX).min(layer.w) {
                    let m = layer.mat[layer.idx(x, y)];
                    total += 1;
                    match table.class(m) {
                        MaterialClass::Powder => counts[0] += 1,
                        MaterialClass::Liquid => counts[1] += 1,
                        MaterialClass::Solid => counts[2] += 1,
                        MaterialClass::Gas => counts[3] += 1,
                        MaterialClass::Empty => {}
                    }
                }
            }
            let slot = &mut pass.cell[cy * cw + cx];
            let filled: u32 = counts.iter().sum();
            if total == 0 || filled == 0 {
                *slot = (EMPTY, 0);
                continue;
            }

            // the dominant class wins the block — this is a diagram, not a blend
            let mut best = 0usize;
            for k in 1..4 {
                if counts[k] > counts[best] {
                    best = k;
                }
            }
            // coverage is the alpha: a half-empty block reads as half-drawn, which is what
            // makes a sculpted funnel still legible once the pixels are gone
            let cov = filled as f32 / total as f32;
            *slot = (best as u8, (cov * 255.0).clamp(0.0, 255.0) as u8);
        }
    }
}

/// 64-bit hash of the `mat` bytes inside `r` — the schematic's only input, so equal hashes
/// mean an equal coarse result.
///
/// Four independent multiply chains, eight bytes at a time, so the loop is not latency-bound
/// on a single dependency. A collision would show as one stale coarse block until that chunk
/// next changed; at 2^-64 per chunk per tick that is not worth a second dirty list.
fn hash_mat(layer: &Layer, r: CellRect) -> u64 {
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut lanes: [u64; 4] = [
        0xcbf2_9ce4_8422_2325,
        0x9e37_79b9_7f4a_7c15,
        0xff51_afd7_ed55_8ccd,
        0xc4ce_b9fe_1a85_ec53,
    ];
    let w = layer.w as usize;
    for y in r.y0..=r.y1 {
        let row = y as usize * w;
        let s = &layer.mat[row + r.x0 as usize..=row + r.x1 as usize];
        let mut words = s.chunks_exact(8);
        let mut i = 0usize;
        for c in &mut words {
            let v = u64::from_le_bytes(c.try_into().unwrap());
            let l = &mut lanes[i & 3];
            *l = (*l ^ v).wrapping_mul(PRIME);
            i += 1;
        }
        for &b in words.remainder() {
            lanes[0] = (lanes[0] ^ b as u64).wrapping_mul(PRIME);
        }
    }
    let mut h =
        lanes[0] ^ lanes[1].rotate_left(16) ^ lanes[2].rotate_left(32) ^ lanes[3].rotate_left(48);
    // splitmix64 finaliser, so a one-byte change reaches every output bit
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

#[inline]
fn mix(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).clamp(0.0, 255.0) as u8
}

/// source-over composite of `(rgb, a)` onto an existing RGBA8 pixel
#[inline]
fn over(dst: &mut [u8], rgb: [u8; 3], a: u8) {
    let sa = a as f32 / 255.0;
    let da = dst[3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        dst.fill(0);
        return;
    }
    for k in 0..3 {
        let s = rgb[k] as f32 * sa;
        let d = dst[k] as f32 * da * (1.0 - sa);
        dst[k] = ((s + d) / out_a).clamp(0.0, 255.0) as u8;
    }
    dst[3] = (out_a * 255.0).clamp(0.0, 255.0) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::asset::Assets;
    use bevy::image::Image;
    use pixelsim::{LayerSlot, MaterialTable};

    fn canvas(images: &mut Assets<Image>, w: u16, h: u16) -> LayerCanvas {
        LayerCanvas::new(
            images,
            w.div_ceil(CELL_PX).max(1),
            h.div_ceil(CELL_PX).max(1),
            false,
            true,
        )
    }

    /// A deliberately awkward layer: not a multiple of `CHUNK_PX`, not a multiple of
    /// `CELL_PX`, and with partly-filled blocks so coverage alpha is exercised.
    fn scene(t: &MaterialTable, w: u16, h: u16) -> Layer {
        let sand = t.id("sand").unwrap().0;
        let water = t.id("water").unwrap().0;
        let mut l = Layer::new(w, h, LayerSlot::Plant, 7);
        for y in 0..h {
            for x in 0..w {
                let m = match (x / 5 + y / 3) % 4 {
                    0 => sand,
                    1 => water,
                    2 => 0,
                    _ => sand,
                };
                let i = l.idx(x, y);
                l.mat[i] = m;
            }
        }
        l
    }

    /// The cache is an efficiency device: a cached composite must be byte-identical to one
    /// baked from a cold cache, both on the first pass and after an edit.
    #[test]
    fn cached_composite_matches_a_cold_one() {
        let t = MaterialTable::embedded();
        let mut images = Assets::<Image>::default();
        let (w, h) = (141u16, 77u16);
        let mut layer = scene(&t, w, h);

        let mut warm = canvas(&mut images, w, h);
        let mut cold = canvas(&mut images, w, h);

        for edit in 0..4u16 {
            if edit > 0 {
                // one cell, deep inside one chunk, in a different chunk each round
                let x = (edit * 37) % w;
                let y = (edit * 23) % h;
                let m = if edit % 2 == 0 {
                    0
                } else {
                    t.id("sand").unwrap().0
                };
                let i = layer.idx(x, y);
                layer.mat[i] = m;
            }

            warm.clear();
            bake_schematic(&mut warm, &layer, &t, 1);

            // a canvas that has never seen this layer before
            let mut fresh = canvas(&mut images, w, h);
            fresh.clear();
            bake_schematic(&mut fresh, &layer, &t, 1);

            assert_eq!(
                warm.rgba, fresh.rgba,
                "cached composite drifted at edit {edit}"
            );
            cold = fresh;
        }
        assert!(!cold.rgba.is_empty());
    }

    /// Two layers in one composite must not share a cache slot, and the back-to-front
    /// order must survive caching.
    #[test]
    fn passes_are_independent_and_ordered() {
        let t = MaterialTable::embedded();
        let mut images = Assets::<Image>::default();
        let (w, h) = (64u16, 32u16);
        let sand = t.id("sand").unwrap().0;
        let water = t.id("water").unwrap().0;

        let mut back = Layer::new(w, h, LayerSlot::Gangway, 1);
        back.mat.fill(sand);
        let mut front = Layer::new(w, h, LayerSlot::Plant, 2);
        front.mat.fill(water);

        let mut c = canvas(&mut images, w, h);
        let mut last = Vec::new();
        for _ in 0..3 {
            c.clear();
            bake_schematic(&mut c, &back, &t, 1);
            bake_schematic(&mut c, &front, &t, 0);
            if !last.is_empty() {
                assert_eq!(last, c.rgba, "composite changed between identical ticks");
            }
            last = c.rgba.clone();
        }

        // fully opaque front layer wins every texel
        let mut expect = c.rgba.clone();
        expect.fill(0);
        let mut solo = canvas(&mut images, w, h);
        solo.clear();
        bake_schematic(&mut solo, &front, &t, 0);
        assert_eq!(
            solo.rgba, last,
            "an opaque front layer must cover the back one"
        );
    }
}
