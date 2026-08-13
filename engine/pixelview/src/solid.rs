//! Turning a layer's occupancy grid into **real 3D geometry**.
//!
//! # What this replaces, and why
//!
//! The first perspective pass drew each slab as a flat quad and *painted* a fake side face
//! along every silhouette edge (`slab.rs`). That gives depth cues but the slabs never
//! connect: there is nothing between the front of one plane and the front of the next, so a
//! hole dug in the Plant shows the slab behind as a picture rather than as a space, and
//! nothing can cast a real shadow on anything.
//!
//! Here each slab is a **solid one z-step deep**, and the stack is contiguous — slab `i`
//! occupies `z ∈ [-(i+1)·D, -i·D]`, so the back face of one slab is the front face of the
//! next. That is LittleBigPlanet's grammar exactly: discrete integer layers, each an
//! extrusion, packed against each other. Dig a hole in the Plant and you are looking down a
//! shaft whose walls you can see, at the Gangway behind it.
//!
//! # The two pieces of geometry
//!
//! **Front face** — one quad per chunk, UV-mapped to the layer's baked canvas. The canvas
//! already carries per-pixel colour, wetness, grain and the depth treatment, and air bakes
//! to zero alpha, so an alpha-masked quad cuts the holes for free. One quad instead of a
//! quad per cell is not a saving worth arguing about at 128 chunks — it is what keeps the
//! per-pixel look pixel-exact, because the texture is doing the shape.
//!
//! **Walls** — a quad wherever a solid cell borders air, spanning the slab's full depth,
//! with an outward normal so the light rig shades it. These are the geometry that makes the
//! stack read as solid, and they are generated per chunk so a dirty chunk rebuilds only its
//! own walls. Runs are merged along the boundary, which turns a 64-cell straight wall into
//! one quad rather than 64.
//!
//! Wall UVs point at the *source* cell, so a wall takes the colour of the material it
//! belongs to and needs no palette of its own — brass walls look like brass, wet sand walls
//! look like wet sand, and it all falls out of the texture that already exists. The
//! corollary, learned the hard way: a merged run may only span **one material**, because the
//! whole quad samples the run's first cell.

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};
use pixelsim::{CHUNK_PX, Layer};

/// One z-step, in sim px. The stack is contiguous, so this is both a slab's depth and the
/// spacing between slabs.
///
/// `PLAYER_PX` is 16, so a slab is a room one and a half player-heights deep — enough that
/// a wall reads at deck scale without the stack becoming a corridor.
pub const SLAB_DEPTH_PX: f32 = 5.0;

/// Vertex buffers under construction.
#[derive(Default)]
pub struct MeshBuf {
    pub pos: Vec<[f32; 3]>,
    pub nrm: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub idx: Vec<u32>,
}

impl MeshBuf {
    pub fn clear(&mut self) {
        self.pos.clear();
        self.nrm.clear();
        self.uv.clear();
        self.idx.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.idx.is_empty()
    }

    /// Push one quad, `a`-`b`-`c`-`d` counter-clockwise seen from outside.
    #[allow(clippy::too_many_arguments)]
    fn quad(
        &mut self,
        a: [f32; 3],
        b: [f32; 3],
        c: [f32; 3],
        d: [f32; 3],
        n: [f32; 3],
        uv: [f32; 2],
    ) {
        let base = self.pos.len() as u32;
        for p in [a, b, c, d] {
            self.pos.push(p);
            self.nrm.push(n);
            // Every vertex of a wall samples the same texel — the cell the wall belongs to.
            // A gradient across the wall would look like the material sliding, and at one
            // texel per cell there is nothing to interpolate toward anyway.
            self.uv.push(uv);
        }
        self.idx
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn into_mesh(&self) -> Mesh {
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.pos.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.nrm.clone())
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uv.clone())
        .with_inserted_indices(Indices::U32(self.idx.clone()))
    }
}

/// Cell -> world. The vessel is centred on the origin, world `+y` is up, cell `y = 0` is the
/// top row. Same convention as `cursor.rs`, which is the only other place that knows it.
#[inline]
fn cell_x(x: f32, w: u16) -> f32 {
    x - w as f32 * 0.5
}

#[inline]
fn cell_y(y: f32, h: u16) -> f32 {
    h as f32 * 0.5 - y
}

/// Texel centre for a cell, in UV space.
#[inline]
fn cell_uv(x: u16, y: u16, w: u16, h: u16) -> [f32; 2] {
    [(x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32]
}

/// The front face of one chunk: a single quad, UV-mapped to that region of the canvas.
///
/// Air is transparent in the bake, so the material's alpha mask cuts the holes — which is
/// also what makes the holes cast correctly shaped shadows.
pub fn build_front(out: &mut MeshBuf, layer: &Layer, chunk: usize, z_front: f32) {
    let Some((x0, y0, x1, y1)) = chunk_rect(layer, chunk) else {
        return;
    };
    let (w, h) = (layer.w, layer.h);
    // any solid cell at all? an all-air chunk needs no quad
    let mut any = false;
    for y in y0..=y1 {
        let row = y as usize * w as usize;
        for x in x0..=x1 {
            if layer.mat[row + x as usize] != 0 {
                any = true;
                break;
            }
        }
        if any {
            break;
        }
    }
    if !any {
        return;
    }

    let (lx, rx) = (cell_x(x0 as f32, w), cell_x(x1 as f32 + 1.0, w));
    let (ty, by) = (cell_y(y0 as f32, h), cell_y(y1 as f32 + 1.0, h));
    let (u0, v0) = (x0 as f32 / w as f32, y0 as f32 / h as f32);
    let (u1, v1) = ((x1 as f32 + 1.0) / w as f32, (y1 as f32 + 1.0) / h as f32);

    let base = out.pos.len() as u32;
    let n = [0.0, 0.0, 1.0];
    for (p, uv) in [
        ([lx, by, z_front], [u0, v1]),
        ([rx, by, z_front], [u1, v1]),
        ([rx, ty, z_front], [u1, v0]),
        ([lx, ty, z_front], [u0, v0]),
    ] {
        out.pos.push(p);
        out.nrm.push(n);
        out.uv.push(uv);
    }
    out.idx
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

/// The walls of one chunk: a quad wherever a solid cell borders air, spanning the slab.
///
/// Runs are merged along each boundary direction, so a straight wall costs one quad however
/// long it is. Boundaries against a *neighbouring chunk's* cells are read across the chunk
/// edge rather than assumed solid — otherwise every chunk would be boxed in by its own
/// walls, which is both wrong and four times the geometry.
pub fn build_walls(out: &mut MeshBuf, layer: &Layer, chunk: usize, z_front: f32, z_back: f32) {
    let Some((x0, y0, x1, y1)) = chunk_rect(layer, chunk) else {
        return;
    };
    let (w, h) = (layer.w, layer.h);
    let mat_at = |x: i32, y: i32| -> u8 {
        if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
            // Outside the vessel is open air, so the outer rim of the world gets walls. That
            // is correct: the vessel's own edge is a cut face and should look like one.
            return 0;
        }
        layer.mat[y as usize * w as usize + x as usize]
    };
    let solid = |x: i32, y: i32| -> bool { mat_at(x, y) != 0 };

    // --- vertical walls: left and right faces, merged down columns -------------
    for x in x0..=x1 {
        for (dx, nx) in [(-1i32, [-1.0f32, 0.0, 0.0]), (1, [1.0, 0.0, 0.0])] {
            let mut y = y0;
            while y <= y1 {
                if !(solid(x as i32, y as i32) && !solid(x as i32 + dx, y as i32)) {
                    y += 1;
                    continue;
                }
                let start = y;
                // Merge only across cells of the *same material*. A run takes its UV from
                // its first cell, so a run that crossed a material boundary would paint the
                // whole quad with the wrong texture — which is exactly what made a water
                // surface next to a sand pile come out sand-coloured.
                let mat = mat_at(x as i32, y as i32);
                while y + 1 <= y1
                    && mat_at(x as i32, y as i32 + 1) == mat
                    && !solid(x as i32 + dx, y as i32 + 1)
                {
                    y += 1;
                }
                let face_x = cell_x(if dx < 0 { x as f32 } else { x as f32 + 1.0 }, w);
                let (top, bot) = (cell_y(start as f32, h), cell_y(y as f32 + 1.0, h));
                let uv = cell_uv(x, start, w, h);
                // wound so the normal faces out of the material on both sides
                if dx < 0 {
                    out.quad(
                        [face_x, bot, z_back],
                        [face_x, bot, z_front],
                        [face_x, top, z_front],
                        [face_x, top, z_back],
                        nx,
                        uv,
                    );
                } else {
                    out.quad(
                        [face_x, bot, z_front],
                        [face_x, bot, z_back],
                        [face_x, top, z_back],
                        [face_x, top, z_front],
                        nx,
                        uv,
                    );
                }
                y += 1;
            }
        }
    }

    // --- horizontal walls: top and bottom faces, merged along rows -------------
    for y in y0..=y1 {
        for (dy, ny) in [(-1i32, [0.0f32, 1.0, 0.0]), (1, [0.0, -1.0, 0.0])] {
            let mut x = x0;
            while x <= x1 {
                if !(solid(x as i32, y as i32) && !solid(x as i32, y as i32 + dy)) {
                    x += 1;
                    continue;
                }
                let start = x;
                let mat = mat_at(x as i32, y as i32);
                while x + 1 <= x1
                    && mat_at(x as i32 + 1, y as i32) == mat
                    && !solid(x as i32 + 1, y as i32 + dy)
                {
                    x += 1;
                }
                // cell y grows downward, so dy < 0 is the *top* face
                let face_y = cell_y(if dy < 0 { y as f32 } else { y as f32 + 1.0 }, h);
                let (left, right) = (cell_x(start as f32, w), cell_x(x as f32 + 1.0, w));
                let uv = cell_uv(start, y, w, h);
                if dy < 0 {
                    out.quad(
                        [left, face_y, z_front],
                        [right, face_y, z_front],
                        [right, face_y, z_back],
                        [left, face_y, z_back],
                        ny,
                        uv,
                    );
                } else {
                    out.quad(
                        [left, face_y, z_back],
                        [right, face_y, z_back],
                        [right, face_y, z_front],
                        [left, face_y, z_front],
                        ny,
                        uv,
                    );
                }
                x += 1;
            }
        }
    }
}

#[inline]
fn chunk_rect(layer: &Layer, chunk: usize) -> Option<(u16, u16, u16, u16)> {
    let cw = layer.chunks.cw as usize;
    let cx = (chunk % cw) as u16;
    let cy = (chunk / cw) as u16;
    let x0 = cx.checked_mul(CHUNK_PX)?;
    let y0 = cy.checked_mul(CHUNK_PX)?;
    if x0 >= layer.w || y0 >= layer.h {
        return None;
    }
    Some((
        x0,
        y0,
        (x0 + CHUNK_PX - 1).min(layer.w - 1),
        (y0 + CHUNK_PX - 1).min(layer.h - 1),
    ))
}

/// `z` of the front and back faces of slab `index`, given a slab depth.
///
/// Contiguous by construction: slab `i`'s back face *is* slab `i+1`'s front face. That is
/// the whole point — the stack is a solid block of layers, not planes floating in space.
#[inline]
pub fn slab_z(index: usize, depth: f32) -> (f32, f32) {
    let front = -(index as f32) * depth;
    (front, front - depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelsim::{LayerSlot, MaterialTable};

    fn block_layer() -> Layer {
        let t = MaterialTable::embedded();
        let stone = t.id("stone").unwrap().0;
        let mut l = Layer::new(128, 128, LayerSlot::Plant, 5);
        for y in 40..60u16 {
            for x in 40..60u16 {
                let i = l.idx(x, y);
                l.mat[i] = stone;
            }
        }
        l
    }

    /// The stack has to be *contiguous*: no gap between one slab's back face and the next
    /// slab's front face, or the layers read as floating planes again.
    #[test]
    fn slabs_abut_with_no_gap() {
        let d = SLAB_DEPTH_PX;
        for i in 0..4 {
            let (front, back) = slab_z(i, d);
            assert!((front - back - d).abs() < 1e-6, "slab {i} is not {d} deep");
            let (next_front, _) = slab_z(i + 1, d);
            assert!(
                (back - next_front).abs() < 1e-6,
                "gap between slab {i} back {back} and slab {} front {next_front}",
                i + 1
            );
        }
    }

    /// A solid block must be walled on all four sides and nowhere else. This is the geometry
    /// that makes a dug hole read as a shaft rather than as a hole in a picture.
    #[test]
    fn a_block_is_walled_on_its_boundary_only() {
        let layer = block_layer();
        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(1, SLAB_DEPTH_PX);
        for c in 0..layer.chunks.len() {
            build_walls(&mut buf, &layer, c, front, back);
        }
        assert!(!buf.is_empty(), "a solid block produced no walls at all");

        // every wall vertex sits on the block's boundary in the axis its normal points along
        let (w, h) = (layer.w, layer.h);
        let (min_x, max_x) = (cell_x(40.0, w), cell_x(60.0, w));
        let (max_y, min_y) = (cell_y(40.0, h), cell_y(60.0, h));
        for (p, n) in buf.pos.iter().zip(&buf.nrm) {
            if n[0].abs() > 0.5 {
                assert!(
                    (p[0] - min_x).abs() < 1e-3 || (p[0] - max_x).abs() < 1e-3,
                    "side wall at x={} is not on the block boundary",
                    p[0]
                );
            }
            if n[1].abs() > 0.5 {
                assert!(
                    (p[1] - min_y).abs() < 1e-3 || (p[1] - max_y).abs() < 1e-3,
                    "top/bottom wall at y={} is not on the block boundary",
                    p[1]
                );
            }
            // and every wall spans exactly the slab's depth
            assert!(
                (p[2] - front).abs() < 1e-3 || (p[2] - back).abs() < 1e-3,
                "wall vertex z={} is neither face of the slab",
                p[2]
            );
        }
    }

    /// Runs must merge, or a 20-cell wall costs 20 quads and a real vessel costs tens of
    /// thousands. Four sides of a 20x20 block is four quads if merging works.
    #[test]
    fn straight_runs_merge_into_single_quads() {
        let layer = block_layer();
        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(0, SLAB_DEPTH_PX);
        for c in 0..layer.chunks.len() {
            build_walls(&mut buf, &layer, c, front, back);
        }
        let quads = buf.idx.len() / 6;
        // the block straddles chunk boundaries, so a few extra seams are expected — but it
        // must be a handful, not 80
        assert!(
            quads <= 12,
            "{quads} quads for a 20x20 block: runs are not merging"
        );
    }

    /// Air must produce nothing at all: an empty chunk is the common case in a dug-out
    /// vessel and it has to be free.
    #[test]
    fn empty_chunks_produce_no_geometry() {
        let t = MaterialTable::embedded();
        let layer = Layer::new(128, 128, LayerSlot::Plant, 1);
        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(0, SLAB_DEPTH_PX);
        for c in 0..layer.chunks.len() {
            build_front(&mut buf, &layer, c, front);
            build_walls(&mut buf, &layer, c, front, back);
        }
        assert!(
            buf.is_empty(),
            "empty layer produced {} verts",
            buf.pos.len()
        );
        let _ = t;
    }

    /// A hole dug in a solid field must be walled on its *inside* — that is what makes the
    /// hole a shaft you can see into rather than a gap in a flat image.
    #[test]
    fn a_hole_is_walled_from_the_inside() {
        let t = MaterialTable::embedded();
        let stone = t.id("stone").unwrap().0;
        let mut layer = Layer::new(128, 128, LayerSlot::Plant, 2);
        layer.fill(stone);
        for y in 50..60u16 {
            for x in 50..60u16 {
                let i = layer.idx(x, y);
                layer.mat[i] = 0;
            }
        }
        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(1, SLAB_DEPTH_PX);
        for c in 0..layer.chunks.len() {
            build_walls(&mut buf, &layer, c, front, back);
        }
        // walls exist, and they face *inward* toward the hole: the +x wall of the hole's left
        // side has a normal pointing +x, which is only produced by material to its left
        let inward = buf
            .nrm
            .iter()
            .zip(&buf.pos)
            .any(|(n, p)| n[0] > 0.5 && (p[0] - cell_x(50.0, layer.w)).abs() < 1e-3);
        assert!(inward, "the hole has no inward-facing wall");
    }
}

#[cfg(test)]
mod material_run_tests {
    use super::*;
    use pixelsim::{LayerSlot, MaterialTable};

    /// Reproduction of a bug spotted on screen: a water surface lying next to a sand pile
    /// came out **sand-coloured**.
    ///
    /// Cause: a top-face run grew rightward while the next cell was merely solid, so a run
    /// starting in sand carried on across the water, and the merged quad sampled its first
    /// cell for the whole span. Runs now only merge across one material.
    #[test]
    fn a_run_never_spans_two_materials() {
        let t = MaterialTable::embedded();
        let sand = t.id("sand").unwrap().0;
        let water = t.id("water").unwrap().0;
        let mut l = Layer::new(64, 64, LayerSlot::Plant, 3);
        // one row, air above: sand on the left half, water on the right
        let y = 30u16;
        for x in 10..20u16 {
            let i = l.idx(x, y);
            l.mat[i] = sand;
        }
        for x in 20..40u16 {
            let i = l.idx(x, y);
            l.mat[i] = water;
        }

        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(1, SLAB_DEPTH_PX);
        for c in 0..l.chunks.len() {
            build_walls(&mut buf, &l, c, front, back);
        }

        // the top faces are the quads whose normal is +y
        let (w, h) = (l.w, l.h);
        let top_y = cell_y(y as f32, h);
        let sand_u = cell_uv(10, y, w, h)[0];
        let water_u = cell_uv(20, y, w, h)[0];

        // Every top-face vertex must sample a texel that belongs to the material actually
        // under it: a vertex spanning into water territory may not carry the sand UV.
        let mut saw_sand = false;
        let mut saw_water = false;
        for (p, (n, uv)) in buf.pos.iter().zip(buf.nrm.iter().zip(&buf.uv)) {
            if n[1] <= 0.5 || (p[1] - top_y).abs() > 1e-3 {
                continue;
            }
            let cell = (uv[0] * w as f32 - 0.5).round() as i32;
            assert!(
                (10..40).contains(&cell),
                "top face sampled cell {cell}, outside the row"
            );
            let expected = if cell < 20 { sand } else { water };
            assert_eq!(
                l.mat_at(cell as u16, y),
                expected,
                "top face UV points at cell {cell}, which is not the material it draws"
            );
            if (uv[0] - sand_u).abs() < 1e-4 {
                saw_sand = true;
            }
            if (uv[0] - water_u).abs() < 1e-4 {
                saw_water = true;
            }
        }
        assert!(saw_sand, "the sand's top face is missing");
        assert!(
            saw_water,
            "the water's top face is missing — it was merged into the sand's run"
        );
    }

    /// The fix must not cost the merge: a single-material wall is still one quad.
    #[test]
    fn same_material_runs_still_merge() {
        let t = MaterialTable::embedded();
        let sand = t.id("sand").unwrap().0;
        let mut l = Layer::new(64, 64, LayerSlot::Plant, 4);
        for x in 8..40u16 {
            let i = l.idx(x, 20);
            l.mat[i] = sand;
        }
        let mut buf = MeshBuf::default();
        let (front, back) = slab_z(0, SLAB_DEPTH_PX);
        for c in 0..l.chunks.len() {
            build_walls(&mut buf, &l, c, front, back);
        }
        let tops = buf.nrm.iter().filter(|n| n[1] > 0.5).count() / 4;
        assert!(tops <= 2, "{tops} quads for one 32-cell top face");
    }
}
