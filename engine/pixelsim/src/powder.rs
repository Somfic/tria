//! Powder solver: fall, diagonal, creep, settle. Exactly one `&mut Layer`.
//! Dispatch is on `MaterialClass` only — never a `match` on a material id.

use crate::cell::FLAG_MOVED;
use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::repose::ReposeLut;
use crate::slurry;
use crate::step::for_awake_cells_bottom_up;

/// grains finer than this entrain into a liquid instead of displacing it (µm).
///
/// Set above sand (400 µm) so sand clouds into suspension and slowly settles back out the
/// same way the finer powders do — `v_settle` already puts sand at ~0.32 cells/tick, a
/// visible ooze rather than a drop. Kept below grit (1200 µm), which is coarse enough to
/// just displace the water and sink.
pub const ENTRAIN_GRAIN_UM: f32 = 500.0;
/// how far up a column is probed when measuring the local slope
pub const COLUMN_PROBE: u16 = 4;

pub fn step_powder(layer: &mut Layer, table: &MaterialTable, lut: &ReposeLut) {
    for_awake_cells_bottom_up(layer, |l, x, y| {
        powder_cell(l, table, lut, x, y);
    });
}

#[inline]
fn powder_cell(l: &mut Layer, table: &MaterialTable, lut: &ReposeLut, x: u16, y: u16) {
    let i = l.idx(x, y);
    let m = l.mat[i];
    if table.class(m) != MaterialClass::Powder {
        return;
    }
    if l.flags[i] & FLAG_MOVED != 0 {
        return;
    }
    l.stats.active_cells += 1;

    let wet = l.wetness[i] as f32 * (1.0 / 255.0);
    let rho = table.wet_density(m, wet);
    let theta = table.wet_repose(m, wet);

    // 1. fall
    if y + 1 < l.h {
        if try_step(l, table, m, rho, wet, x, y, x, y + 1) {
            return;
        }
        // sieving: a solid whose aperture admits this grain is passed through, not
        // occupied — the grain lands on the far side
        let below = l.mat_at(x, y + 1);
        if table.is_solid(below)
            && table.passes_powder(below, m)
            && y + 2 < l.h
            && try_step(l, table, m, rho, wet, x, y, x, y + 2)
        {
            return;
        }
    }

    // 2. diagonal, unless cohesion holds it (steep, wet piles stick)
    let stuck = theta > 45.0 && l.rng.chance_u8(lut.p_stick(theta));
    if !stuck && y + 1 < l.h {
        let dxs = if l.rng.bit() { [-1i32, 1] } else { [1, -1] };
        for dx in dxs {
            let nx = x as i32 + dx;
            if nx < 0 || nx >= l.w as i32 {
                continue;
            }
            let nx = nx as u16;
            // the lateral cell must not be a wall, or powder leaks through a 1-px
            // diagonal seam
            let side = l.mat_at(nx, y);
            if table.is_solid(side) && !table.passes_powder(side, m) {
                continue;
            }
            if try_step(l, table, m, rho, wet, x, y, nx, y + 1) {
                return;
            }
        }
    }

    // 3. creep — this is what pulls a pile below 45 degrees.
    //
    // A 45-degree staircase is stable under rules 1 and 2 alone: every surface grain
    // has its diagonal blocked by the next column. Creep is the avalanche that breaks
    // it: a *surface* grain steps one cell sideways when the surface `span` columns
    // downhill is at least `drop` cells lower. The geometry is the stop condition, so
    // the pile comes to rest instead of flattening; `p_creep` decides how much of the
    // relaxation actually happens before the pile is buried.
    if theta < 45.0 {
        let p = lut.p_creep(theta);
        if p == 0 {
            return;
        }
        // buried grains do not avalanche
        if y > 0 && !table.is_empty(l.mat_at(x, y - 1)) {
            return;
        }
        let span = lut.creep_span(theta) as i32;
        let drop = lut.creep_drop(theta);
        let dxs = if l.rng.bit() { [-1i32, 1] } else { [1, -1] };
        for dx in dxs {
            let nx = x as i32 + dx;
            let fx = x as i32 + dx * span;
            if nx < 0 || nx >= l.w as i32 || fx < 0 || fx >= l.w as i32 {
                continue;
            }
            let nx = nx as u16;
            if !table.is_empty(l.mat_at(nx, y)) {
                continue;
            }
            // strictly deeper than `drop`, so the pile comes to rest at exactly
            // atan(drop / span) rather than relaxing one step past it
            if surface_drop(l, table, fx as u16, y, drop + 1) <= drop {
                continue;
            }
            if l.rng.chance_u8(p) {
                l.commit_move((x, y), (nx, y));
                l.stats.dust += 1.0 - wet;
            }
            return;
        }
    }
    // 4. settle: nothing to do — ChunkMap::end_tick owns sleeping
}

/// Attempts one move of the powder at `(x, y)` into `(tx, ty)`. Returns whether the
/// grain moved (or was entrained, which also retires the cell).
#[inline]
#[allow(clippy::too_many_arguments)]
fn try_step(
    l: &mut Layer,
    table: &MaterialTable,
    m: u8,
    rho: f32,
    wet: f32,
    x: u16,
    y: u16,
    tx: u16,
    ty: u16,
) -> bool {
    let ti = l.idx(tx, ty);
    let t = l.mat[ti];

    if table.is_empty(t) {
        l.commit_move((x, y), (tx, ty));
        l.stats.dust += 1.0 - wet;
        return true;
    }

    // displaceable: a lighter liquid gets swapped aside, unless the grain is fine
    // enough to go into suspension instead
    if table.class(t) == MaterialClass::Liquid && table.density(t) < rho {
        if table.grain(m) < ENTRAIN_GRAIN_UM && slurry::can_entrain(l, m, ti) {
            slurry::entrain(l, table, (x, y), (tx, ty));
            return true;
        }
        l.commit_move((x, y), (tx, ty));
        l.coarse.note_flow(tx, ty);
        return true;
    }

    false
}

/// Number of consecutive non-empty cells at and above `(x, y)`, capped at
/// [`COLUMN_PROBE`]. The local pile depth.
#[inline]
pub fn column_height(l: &Layer, table: &MaterialTable, x: u16, y: u16) -> u16 {
    let mut n = 0u16;
    let mut yy = y as i32;
    while n < COLUMN_PROBE && yy >= 0 {
        if table.is_empty(l.mat_at(x, yy as u16)) {
            break;
        }
        n += 1;
        yy -= 1;
    }
    n
}

/// How far below row `y` the surface of column `x` sits, capped at `cap`. `0` means
/// the surface is level with `y` (or above it). The layer's bottom edge counts as
/// support, so a lone grain on the floor sees no drop and never creeps.
#[inline]
pub fn surface_drop(l: &Layer, table: &MaterialTable, x: u16, y: u16, cap: u16) -> u16 {
    let mut d = 0u16;
    let mut yy = y as u32;
    while d < cap {
        if yy >= l.h as u32 || !table.is_empty(l.mat_at(x, yy as u16)) {
            break;
        }
        d += 1;
        yy += 1;
    }
    d
}
