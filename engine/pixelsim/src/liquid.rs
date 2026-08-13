//! Liquid solver: fall, buoyant swap, lateral dispersion, head equalisation.
//! `head` never crosses layers.

use crate::cell::{Bitset, FLAG_MOVED};
use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::step::{chunk_span, for_awake_cells_bottom_up};

/// a cell only spreads sideways under pressure — it needs at least this much liquid
/// stacked above it. Without the threshold a lone droplet wanders forever and its
/// chunk never sleeps.
pub const SPREAD_MIN_HEAD: u8 = 2;

/// Sets `head` for every awake liquid cell and 0 everywhere else.
///
/// `head` is *not* simply this cell's own depth. It is
/// `max(head_above + 1, head of the liquid cells left and right on this row)`, which
/// makes it the deepest column the cell is laterally connected to along its own row.
/// That single extra max is what turns a purely local rule into level equalisation: a
/// shallow cell next to a deep body inherits the body's head and therefore spreads,
/// and once the surface is flat every surface cell is back to `head = 1` and the pool
/// goes quiet. Without it, a 1-cell surface staircase is a fixed point and a pool
/// settles as a wedge.
///
/// Three sweeps per row: down + left-to-right, then right-to-left. It executes before
/// the move pass, so `head` is one tick stale relative to moves — accepted, and
/// required for determinism.
pub fn compute_head(layer: &mut Layer, table: &MaterialTable) {
    let active = core::mem::replace(&mut layer.chunks.active, Bitset::new(0));
    let (cw, ch) = (layer.chunks.cw, layer.chunks.ch);
    let (w, h) = (layer.w, layer.h);
    let stride = w as usize;

    for cy in 0..ch {
        let (y0, y1) = chunk_span(cy, h);
        for y in y0..=y1 {
            for cx in 0..cw {
                if !active.get(cy as usize * cw as usize + cx as usize) {
                    continue;
                }
                let (x0, x1) = chunk_span(cx, w);
                for x in x0..=x1 {
                    let i = y as usize * stride + x as usize;
                    if table.class(layer.mat[i]) != MaterialClass::Liquid {
                        layer.head[i] = 0;
                        continue;
                    }
                    let above = if y > 0 { layer.head[i - stride] } else { 0 };
                    let mut hd = above.saturating_add(1);
                    if x > 0 && table.class(layer.mat[i - 1]) == MaterialClass::Liquid {
                        hd = hd.max(layer.head[i - 1]);
                    }
                    layer.head[i] = hd;
                }
            }
            for cx in (0..cw).rev() {
                if !active.get(cy as usize * cw as usize + cx as usize) {
                    continue;
                }
                let (x0, x1) = chunk_span(cx, w);
                for x in (x0..=x1).rev() {
                    let i = y as usize * stride + x as usize;
                    if table.class(layer.mat[i]) != MaterialClass::Liquid {
                        continue;
                    }
                    if x + 1 < w && table.class(layer.mat[i + 1]) == MaterialClass::Liquid {
                        layer.head[i] = layer.head[i].max(layer.head[i + 1]);
                    }
                }
            }
        }
    }

    layer.chunks.active = active;
}

pub fn step_liquid(layer: &mut Layer, table: &MaterialTable) {
    for_awake_cells_bottom_up(layer, |l, x, y| {
        liquid_cell(l, table, x, y);
    });
}

#[inline]
fn liquid_cell(l: &mut Layer, table: &MaterialTable, x: u16, y: u16) {
    let i = l.idx(x, y);
    let m = l.mat[i];
    if table.class(m) != MaterialClass::Liquid {
        return;
    }
    if l.flags[i] & FLAG_MOVED != 0 {
        return;
    }
    l.stats.active_cells += 1;
    let rho = table.density(m);

    // 1. fall, and 2. buoyant swap
    if y + 1 < l.h {
        let below = l.mat_at(x, y + 1);
        if table.is_empty(below) {
            l.commit_move((x, y), (x, y + 1));
            l.coarse.note_flow(x, y + 1);
            return;
        }
        // Percolation: a porous solid is passed through, not occupied — but only if what
        // the water is *carrying* fits through it too.
        //
        // Without the suspension test an aperture filters only what falls on it dry, which
        // is not what a filter does. Suspended solids ride as passengers of a liquid cell,
        // so the sieve was asking the cell whether it fit and never asking the grain: with a
        // 10 µm screen that should retain every 60 µm grain of gold dust, 44 of 98 gold cells
        // still washed through.
        //
        // Refusing the move is all that is needed, because the slurry rules take it from
        // there. A laden cell that cannot descend is at its bed, so `settle_at_bed`
        // concentrates it — bed-load creep pulling neighbouring loads together — until it
        // holds a whole grain and deposits above the screen. That is cake filtration, and it
        // falls out of rules that were already there. A partial load simply waits, which is
        // correct: grains are indivisible here, so there is nothing to deposit yet.
        if table.is_solid(below)
            && table.passes_liquid(below)
            && y + 2 < l.h
            && table.is_empty(l.mat_at(x, y + 2))
            && (l.susp_conc[i] == 0 || table.passes_powder(below, l.susp_mat[i]))
        {
            l.commit_move((x, y), (x, y + 2));
            l.coarse.note_flow(x, y + 2);
            return;
        }
        // denser liquid sinks: swap with the lighter one below
        if table.class(below) == MaterialClass::Liquid && table.density(below) < rho {
            l.commit_move((x, y), (x, y + 1));
            l.coarse.note_flow(x, y + 1);
            return;
        }
    }

    // thick liquids (dispersion 0) only creep, and only sometimes
    let disp = table.dispersion(m);
    if disp == 0 {
        let p = (table.p_move(m) * 255.0) as u8;
        if !l.rng.chance_u8(p) {
            return;
        }
    }

    // 3. diagonal — run down a slope while staying on it.
    //
    // Powder has had this since the start; liquid went straight from "fall" to "lateral
    // dispersion", which cannot follow a surface, and the artefact was loud. Dispersion
    // takes the *furthest* reachable cell with somewhere to fall, so a stream landing on
    // the apex of a pile did not run down the sand — each arriving cell leapt the full
    // `disp` in one tick to the same x and then fell straight, alternating side on tick
    // parity. A steady one-cell stream therefore rendered as two symmetric dotted
    // columns of droplets in mid-air, spaced exactly two rows apart (one per side per
    // two ticks), skiing past the slope without ever touching it.
    //
    // Trying the diagonal first means the cell steps onto the next step down and stays in
    // contact, which is a rivulet. Side order is random rather than tick-parity: identical
    // deterministic trajectories are what turned a stream into a stroboscope, and it is
    // also how `step_powder` does it. The lateral cell must not be a wall, or water leaks
    // through a 1-px diagonal seam — the same guard, for the same reason.
    if y + 1 < l.h {
        let dxs = if l.rng.bit() { [-1i32, 1] } else { [1, -1] };
        for dx in dxs {
            let nx = x as i32 + dx;
            if nx < 0 || nx >= l.w as i32 {
                continue;
            }
            let nx = nx as u16;
            let side = l.mat_at(nx, y);
            if !passable_for_liquid(table, side) {
                continue;
            }
            if table.is_empty(l.mat_at(nx, y + 1)) {
                l.commit_move((x, y), (nx, y + 1));
                l.coarse.note_flow(nx, y + 1);
                return;
            }
        }
    }

    // 4. lateral dispersion — scan the tick's direction first, then the other side,
    // and take the furthest cell that has somewhere to fall. This is what makes
    // water race along a floor instead of stepping one cell per tick.
    let primary: i32 = if l.tick & 1 == 0 { 1 } else { -1 };
    for dir in [primary, -primary] {
        let mut downhill: Option<u16> = None;
        for k in 1..=disp as i32 {
            let nx = x as i32 + dir * k;
            if nx < 0 || nx >= l.w as i32 {
                break;
            }
            let nx = nx as u16;
            let at = l.mat_at(nx, y);
            if !passable_for_liquid(table, at) {
                break;
            }
            // the walk may pass through a porous solid, but the destination must be
            // genuinely empty — a liquid never occupies a mesh cell
            if table.is_empty(at) && y + 1 < l.h && table.is_empty(l.mat_at(nx, y + 1)) {
                downhill = Some(nx);
            }
        }
        if let Some(nx) = downhill {
            l.commit_move((x, y), (nx, y));
            l.coarse.note_flow(nx, y);
            return;
        }
    }

    // 4. head equalisation. A cell with liquid stacked on top spreads into an
    // adjacent empty cell; this is the O(1)/cell subset of the eventual connected
    // component pass, and it is what levels a pool.
    if l.head[i] < SPREAD_MIN_HEAD {
        return;
    }
    for dir in [primary, -primary] {
        let nx = x as i32 + dir;
        if nx < 0 || nx >= l.w as i32 {
            continue;
        }
        let nx = nx as u16;
        let ni = l.idx(nx, y);
        if table.is_empty(l.mat[ni]) && l.head[ni] + SPREAD_MIN_HEAD <= l.head[i] {
            l.commit_move((x, y), (nx, y));
            l.coarse.note_flow(nx, y);
            return;
        }
    }
}

/// empty, or a solid whose aperture passes liquid (percolation)
#[inline]
pub fn passable_for_liquid(table: &MaterialTable, m: u8) -> bool {
    table.is_empty(m) || table.passes_liquid(m)
}
