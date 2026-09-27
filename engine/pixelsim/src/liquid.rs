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

/// The minimum difference, in rows, between the tallest and shortest column of a
/// connected body before [`equalize_levels`] will move a cell between them. Mirrors
/// [`SPREAD_MIN_HEAD`]: a one-cell staircase is a fixpoint, so a settled body makes no
/// move, goes still and sleeps.
pub const LEVEL_MIN_DROP: u16 = 2;

/// How much of a body's surface imbalance is closed each tick, as a divisor: the pass
/// makes at most `1 + surface_columns / LEVEL_RATE_DIV` transfers per body per tick.
///
/// This is the flow-rate dial, and the reason it exists. Water rising up the far arm of a
/// tube is not a local step any falling-sand rule can take — the honest version is a
/// pressure field, which this engine does not have — so equalisation is done by relocating
/// cells across the connected body. Relocating *all* the surplus at once reads as an
/// instant snap with no sense of the water travelling; throttling it to a few cells a tick
/// lets the local down/lateral rules carry water visibly through the connecting channel
/// while the arms creep towards each other over a second or so. Lower = faster/snappier,
/// higher = slower/oozier.
pub const LEVEL_RATE_DIV: usize = 12;

/// Communicating vessels. `liquid_cell` can only ever push water down or level — it has no
/// rule that moves a cell *up* — so water can never climb the far arm of a U-bend and a row
/// of connected columns settles at wildly different heights. This is the connected-component
/// pass the `head` doc-comment anticipates: it floods each connected liquid body, measures
/// the free surface of every column it spans, and relocates cells from its tallest columns
/// onto its shortest until the surface is level.
///
/// It is deliberately *throttled* (see [`LEVEL_RATE_DIV`]): a real pressure solver would
/// push water up the far arm cell by cell, but this engine has none, so the upward step is a
/// cross-body relocation. Doing only a few per tick keeps the level change gradual — the
/// visible flow through the channel is the local rules' doing, and this pass just trues up
/// the far surface behind it. It stays safe on every count the engine cares about:
///
/// * **Mass** — every transfer is a [`Layer::commit_move`] swap; nothing is minted or lost
///   and the aux channels ride along.
/// * **No oscillation** — a cell only ever leaves the very top of one column and lands on the
///   very top of another, so no interior gap is opened for gravity to undo next tick.
/// * **Termination / sleep** — a column is touched at most once per tick and only when the
///   tallest/shortest gap is at least [`LEVEL_MIN_DROP`]; once level nothing moves and the
///   body sleeps.
///
/// Bodies are seeded from awake cells only, so a settled body costs a popcount and is skipped
/// whole; the flood itself crosses the sleep line so a body straddling it equalises as one.
pub fn equalize_levels(layer: &mut Layer, table: &MaterialTable) {
    let (w, h) = (layer.w, layer.h);
    if w == 0 || h == 0 || layer.chunks.active.count() == 0 {
        return;
    }
    let stride = w as usize;
    let n = stride * h as usize;

    let mut visited = Bitset::new(n);
    // scratch, cleared and reused across every body in this pass
    let mut stack: Vec<usize> = Vec::new();
    let mut body: Vec<usize> = Vec::new();
    let mut col_surf: Vec<u16> = vec![u16::MAX; stride];
    let mut touched: Vec<u16> = Vec::new();
    let mut cols: Vec<(u16, u16)> = Vec::new();
    let mut moves: Vec<(usize, usize)> = Vec::new();

    // `active` is lifted out so the loop can call `commit_move(&mut layer)` while walking
    // this tick's awake set — the same borrow trick `for_awake_cells_bottom_up` uses.
    let active = core::mem::replace(&mut layer.chunks.active, Bitset::new(0));
    for c in active.iter_set() {
        let r = layer.chunks.bounds(c, w, h);
        for y in r.y0..=r.y1 {
            for x in r.x0..=r.x1 {
                let seed = y as usize * stride + x as usize;
                if visited.get(seed) || table.class(layer.mat[seed]) != MaterialClass::Liquid {
                    continue;
                }

                // ---- flood the whole connected liquid body (4-connectivity) ----
                body.clear();
                stack.clear();
                stack.push(seed);
                visited.set(seed);
                while let Some(i) = stack.pop() {
                    body.push(i);
                    let cx = i % stride;
                    let cy = i / stride;
                    let mut visit = |ni: usize| {
                        if !visited.get(ni) && table.class(layer.mat[ni]) == MaterialClass::Liquid {
                            visited.set(ni);
                            stack.push(ni);
                        }
                    };
                    if cx > 0 {
                        visit(i - 1);
                    }
                    if cx + 1 < stride {
                        visit(i + 1);
                    }
                    if cy > 0 {
                        visit(i - stride);
                    }
                    if cy + 1 < h as usize {
                        visit(i + stride);
                    }
                }

                // ---- free surface of every column the body spans ----
                // A column's surface is its topmost body cell that has air directly
                // above; a cell capped by solid is under pressure, not a free surface,
                // and takes no part in levelling.
                touched.clear();
                for &i in &body {
                    if i < stride {
                        continue; // top row: no cell above to be the free face
                    }
                    if !table.is_empty(layer.mat[i - stride]) {
                        continue;
                    }
                    let x = (i % stride) as u16;
                    let ycell = (i / stride) as u16;
                    if col_surf[x as usize] == u16::MAX {
                        touched.push(x);
                    }
                    col_surf[x as usize] = col_surf[x as usize].min(ycell);
                }

                cols.clear();
                for &x in &touched {
                    cols.push((col_surf[x as usize], x));
                    col_surf[x as usize] = u16::MAX; // reset for the next body
                }
                // tallest first (smallest surface row), ties by x for determinism
                cols.sort_unstable();

                // ---- pair tallest donor with shortest receiver, inward, throttled ----
                let budget = 1 + cols.len() / LEVEL_RATE_DIV;
                let mut i = 0usize;
                let mut j = cols.len();
                while j > 0 && i + 1 < j && moves.len() < budget {
                    j -= 1;
                    let (surf_hi, x_hi) = cols[i];
                    let (surf_lo, x_lo) = cols[j];
                    // sorted, so once the outermost gap closes every inner gap has too
                    if surf_lo < surf_hi + LEVEL_MIN_DROP {
                        break;
                    }
                    let from = surf_hi as usize * stride + x_hi as usize;
                    let to = (surf_lo - 1) as usize * stride + x_lo as usize;
                    moves.push((from, to));
                    i += 1;
                }

                for &(from, to) in &moves {
                    let f = ((from % stride) as u16, (from / stride) as u16);
                    let t = ((to % stride) as u16, (to / stride) as u16);
                    layer.commit_move(f, t);
                    layer.coarse.note_flow(t.0, t.1);
                    visited.set(to); // it is now body liquid; do not re-seed it
                }
                moves.clear();
            }
        }
    }

    layer.chunks.active = active;
}
