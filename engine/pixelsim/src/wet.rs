//! Wet-material coupling: absorption, gravity drainage, capillary spread, evaporation +
//! stain, drip, and the weakening of things that should not have got wet. Derived
//! properties (`wet_density`, `wet_repose`) are not stored — the powder solver calls them
//! inline.
//!
//! Who can hold water is decided in one place, [`absorbent`]: porous powders, plus any
//! material with an authored `absorbency`, which is how a *solid* like cardboard opts in.
//! Everything below is written against that predicate rather than against
//! `MaterialClass::Powder`, so a soaked box drains, wicks, dries and stains by the same
//! rules as a soaked bed of sand.
//!
//! # The transports, and which one does the work
//!
//! * **Absorption** consumes a liquid cell at a contact face and injects
//!   [`ABSORB_UNITS`] of wetness into the absorbent patch behind it.
//! * **Drainage** ([`drain_cell`]) moves wetness *down* under gravity, from any cell
//!   holding more than its own [`field_capacity`] into the absorbent cell below. This is
//!   what makes a bed drink a pool: it empties the contact patch downward so absorption has
//!   somewhere to put the next cell.
//! * **Sogginess** ([`soggy_cell`]) and **sag** ([`sag_cell`]) are the rules that are not
//!   transport: a solid that stays soaked accumulates [`Layer::wear`], loses its stiffness,
//!   and droops into whatever gap is under it. Nothing is ever destroyed — a wet box
//!   deforms, it does not evaporate.
//! * **Capillary** spread diffuses what is left, biased upward, and is the wicking
//!   effect — it moves water *against* the gradient's easy direction and so needs a
//!   dead-band ([`CAPILLARY_MIN_DELTA`]) to stop it jittering.
//! * **Evaporation** and **drip** return water to the world or the air.
//!
//! Drainage was missing, and its absence is worth stating because the symptom did not
//! look like a missing rule. A pool on dry sand wetted the few cells it touched and then
//! stopped: capillary spread reaches equilibrium once every step of the gradient is under
//! the dead-band, which is a shallow damp patch, and with the patch full, absorption's
//! capacity guard refused every further cell. The bed had 6x the capacity it had used, and
//! the pool sat on top of it for ever. Diffusion with an upward bias is a model of
//! wicking, not of water going downhill.
//!
//! # Why these rules do not ride the awake set
//!
//! Chunk sleep is a *motion* predicate. A settled pool on settled sand does not move, so
//! it sleeps, while still being the exact situation in which every rule here has work to
//! do. All of them therefore run off the **damp** set — chunks that may hold wetness or
//! liquid — which is pruned as it is scanned, so a dry world converges to no wet cost at
//! all. Evaporation and drip were moved off the awake set for this reason once already;
//! absorption and capillary were left behind, and only their capacity stall hid it.

use crate::cell::FLAG_MOVED;
use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::rng::hash_rng;
use crate::step::{chunk_span, for_awake_cells_bottom_up};
use crate::units::AUX_EVERY;

/// wetness units injected by one absorbed water cell
pub const ABSORB_UNITS: u32 = 896;
/// how often a contact face absorbs, in ticks
pub const ABSORB_EVERY: u64 = 4;
/// capillary flow only starts above this wetness gradient
pub const CAPILLARY_MIN_DELTA: i32 = 24;
/// Wetness a cell holds against gravity — the field capacity of the medium.
///
/// Above it, water drains downward; at or below it, water stays put and only capillary
/// spread can move it. 48/255 is about 19% of saturation, so a column that has drained
/// through stays visibly damp while the water it passed on collects at the bottom.
pub const FIELD_CAPACITY: u8 = 48;
/// Most wetness one cell drains downward in a tick.
///
/// A liquid cell is [`ABSORB_UNITS`] of wetness, so this is a wetting front moving about
/// one row per 14 ticks under saturated flow — a quarter of a second per pixel, which is
/// seeping rather than pouring.
pub const DRAIN_MAX_TRANSFER: u8 = 64;
/// holds the wetting front to about 0.25 px/tick
pub const CAPILLARY_MAX_TRANSFER: u8 = 64;
/// ticks per unit of evaporation from an exposed wet cell
pub const EVAPORATE_EVERY: u64 = 40;
/// dirt added when a cell finishes drying — the stain ring
pub const STAIN_STEP: u8 = 24;
/// a cell this wet with air below will eventually drip
pub const DRIP_MIN_WETNESS: u8 = 200;
/// ticks between drips from one cell
pub const DRIP_EVERY: u64 = 120;
/// Wetness at which an absorbent solid starts to lose its strength — half saturation.
///
/// Below it a damp box is merely damp. Cardboard authored at `absorbency = 0.9` reaches
/// this within a few absorption windows of standing in water.
pub const SOG_MIN_WETNESS: u8 = 128;
/// ticks per [`SOG_WEAR_STEP`] of weakening in a soaked solid
pub const SOG_EVERY: u64 = 40;
/// Wear added per [`SOG_EVERY`] ticks to a soaked solid; [`SAG_MIN_WEAR`] is where it
/// starts to deform and 255 is as soft as it gets.
///
/// 8 per 40 ticks is 32 windows, so 1280 ticks — about 21 seconds at 60 Hz — from first
/// soaking to fully soft. Long enough to notice the box darkening and get the contents out,
/// short enough to be a real deadline. Scaled by `absorbency`, so a material that soaks
/// grudgingly also softens slowly.
pub const SOG_WEAR_STEP: u8 = 8;
/// Wear at which a soaked solid has lost enough stiffness to start drooping.
///
/// Half of full wear, so `SOG_EVERY * (SAG_MIN_WEAR / SOG_WEAR_STEP)` — about 640 ticks, 11
/// seconds — of standing in water before the board visibly moves. Sound board above this
/// still pins its neighbours up, so it doubles as the definition of "sound".
pub const SAG_MIN_WEAR: u8 = 128;
/// ticks between one cell's droops, so a soft sheet creeps down instead of falling
pub const SAG_EVERY: u64 = 40;

// the aux pass only runs on multiples of AUX_EVERY, so a rule whose period is not a
// multiple of it would fire at the wrong average rate
const _: () = assert!(EVAPORATE_EVERY % AUX_EVERY == 0);
const _: () = assert!(DRIP_EVERY % AUX_EVERY == 0);
const _: () = assert!(SOG_EVERY % AUX_EVERY == 0);
const _: () = assert!(SAG_EVERY % AUX_EVERY == 0);

/// Per-layer salt for the stateless per-cell hashes, so two layers do not absorb,
/// evaporate and drip in lockstep.
#[inline]
fn salt(layer: &Layer) -> u64 {
    (layer.slot as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF1)
}

/// Absorption, drainage and capillary spread, every tick, over the **damp** set.
///
/// Rows run bottom-up within a chunk so that a column drains into space the cell below it
/// has already given up this tick, which is what lets a wetting front advance at the rate
/// [`DRAIN_MAX_TRANSFER`] states rather than a row per pass.
///
/// The set is cleared per chunk and re-set if the chunk held any wetness or any liquid, so
/// it prunes itself: a vessel with no water in it pays nothing here. A chunk that keeps
/// finding nothing to move drops to the idle cadence rather than being pruned — see
/// [`ChunkMap::wet_idle`]. And see the module docs for why none of this can ride the awake
/// set.
pub fn step_wet(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let s = salt(layer);
    let (cw, ch) = (layer.chunks.cw, layer.chunks.ch);
    let (w, h) = (layer.w, layer.h);
    let todo: Vec<usize> = layer.chunks.damp.iter_set().collect();
    for c in todo {
        let (cx, cy) = ((c % cw as usize) as u16, (c / cw as usize) as u16);
        if cy >= ch {
            continue;
        }
        if layer.chunks.wet_idle(c, tick) {
            continue;
        }
        let (x0, x1) = chunk_span(cx, w);
        let (y0, y1) = chunk_span(cy, h);
        layer.chunks.damp.clear(c);
        let mut work = false;
        let mut moved = false;
        for y in (y0..=y1).rev() {
            for x in x0..=x1 {
                let i = layer.idx(x, y);
                match table.class(layer.mat[i]) {
                    // A pool is work even when it is not moving: it is what absorption
                    // consumes, so its chunk has to stay in the set.
                    MaterialClass::Liquid => {
                        work = true;
                        moved |= absorb_cell(layer, table, s, tick, x, y, i);
                    }
                    // A wet solid drains and wicks exactly like a wet powder does — water
                    // runs down through soaked cardboard and climbs its dry edges — so the
                    // two share the arm. The `absorbent` test keeps stone and tin out.
                    MaterialClass::Powder | MaterialClass::Solid => {
                        if !absorbent(table, layer.mat[i]) {
                            continue;
                        }
                        // A dry grain sitting in water soaks from the side it touches, not
                        // just from a pool draining down onto it or a deposit landing on
                        // top. `absorb_cell` runs from the *water* side and skips cloudy
                        // water, so a grain that sank through a suspension cloud used to
                        // stay bone dry until the dust settled onto it — the "only wet once
                        // dust touches it" report. This wets it from contact instead.
                        if layer.wetness[i] == 0 {
                            if soak_submerged(layer, table, x, y) {
                                work = true;
                                moved = true;
                            }
                            continue;
                        }
                        work = true;
                        moved |= drain_cell(layer, table, x, y, i);
                        moved |= capillary_cell(layer, table, x, y, i);
                    }
                    _ => {}
                }
            }
        }
        if work {
            layer.chunks.damp.set(c);
            layer.chunks.wet_worked(c, moved);
        }
    }
}

/// The low-rate aux pass: evaporation and drip, over the *damp* chunk set, every
/// [`AUX_EVERY`] ticks. Call it from the tick driver when `tick % AUX_EVERY == 0`.
///
/// Why this is not part of [`step_wet`]: chunk sleep is a motion predicate with an
/// 8-tick fuse, and both of these rules have periods of 40 and 120 ticks. A pile that
/// settles is asleep long before its first evaporation window comes round, so hanging
/// them off the awake set stops them for good — a soaked pile stayed at full wetness
/// forever and the drying/stain interaction was unreachable code.
///
/// The damp set is what makes this cheap: it holds only chunks that may contain
/// wetness, is pruned here (every chunk visited and found dry loses its bit), and an
/// idle rock world therefore converges to an empty damp set and zero aux cost. The
/// pass deliberately does not `wake` anything for a wetness change, or the sleep win
/// would be spent re-waking wet chunks every eight ticks.
pub fn step_wet_aux(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let s = salt(layer);
    let (cw, ch) = (layer.chunks.cw, layer.chunks.ch);
    let (w, h) = (layer.w, layer.h);
    // snapshot: the scan itself can mark new chunks damp (a drip lands elsewhere), and
    // those must survive to the next pass rather than being pruned by this one
    let todo: Vec<usize> = layer.chunks.damp.iter_set().collect();
    for c in todo {
        let (cx, cy) = ((c % cw as usize) as u16, (c / cw as usize) as u16);
        if cy >= ch {
            continue;
        }
        let (x0, x1) = chunk_span(cx, w);
        let (y0, y1) = chunk_span(cy, h);
        layer.chunks.damp.clear(c);
        let mut any_wet = false;
        for y in (y0..=y1).rev() {
            for x in x0..=x1 {
                let i = layer.idx(x, y);
                if layer.wetness[i] == 0 || !absorbent(table, layer.mat[i]) {
                    continue;
                }
                any_wet = true;
                soggy_cell(layer, table, s, tick, x, y, i);
                // Sagging moves the cell, so the rules below would be looking at whatever
                // is now sitting in its place — and the cell itself has already been
                // visited, since the scan runs bottom-up and it moved down.
                if sag_cell(layer, table, s, tick, x, y, i) {
                    continue;
                }
                evaporate_cell(layer, table, s, tick, x, y, i);
                if layer.wetness[i] > 0 {
                    drip_cell(layer, table, s, tick, x, y, i);
                }
            }
        }
        if any_wet {
            layer.chunks.damp.set(c);
        }
    }
}

pub fn absorb(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let s = salt(layer);
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        if table.class(l.mat[i]) == MaterialClass::Liquid {
            absorb_cell(l, table, s, tick, x, y, i);
        }
    });
}

/// Single-rule entry point over awake cells, for callers and tests that want gravity
/// drainage without capillary diffusion mixed into the result.
pub fn drain(layer: &mut Layer, table: &MaterialTable) {
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        if table.class(l.mat[i]) == MaterialClass::Powder && l.wetness[i] > 0 {
            drain_cell(l, table, x, y, i);
        }
    });
}

pub fn capillary(layer: &mut Layer, table: &MaterialTable) {
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        if table.class(l.mat[i]) == MaterialClass::Powder && l.wetness[i] > 0 {
            capillary_cell(l, table, x, y, i);
        }
    });
}

/// Single-rule entry point over awake cells. The aux rules are scheduled on the
/// [`AUX_EVERY`] grid, so `tick` is quantised the same way here as in [`step_wet_aux`]:
/// a caller invoking this every tick sees one evaporation window spread over the
/// `AUX_EVERY` ticks of a slot, not `AUX_EVERY` separate windows.
pub fn evaporate(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let s = salt(layer);
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        if table.class(l.mat[i]) == MaterialClass::Powder && l.wetness[i] > 0 {
            evaporate_cell(l, table, s, tick, x, y, i);
        }
    });
}

/// Single-rule entry point over awake cells; see [`evaporate`] on tick quantisation.
pub fn drip(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let s = salt(layer);
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        if table.class(l.mat[i]) == MaterialClass::Powder && l.wetness[i] > 0 {
            drip_cell(l, table, s, tick, x, y, i);
        }
    });
}

/// `true` on the aux slot in which the cell at `(x, y)` is due, for a rule of period
/// `every` ticks. `every / AUX_EVERY` slots make up one period and the cell's hash
/// picks which, so the population still fires uniformly spread in time at exactly the
/// documented average rate — one unit per `every` ticks — despite the pass itself only
/// running every `AUX_EVERY` ticks.
#[inline]
fn due(s: u64, k: u64, tick: u64, x: u16, y: u16, every: u64) -> bool {
    let slots = (every / AUX_EVERY).max(1);
    let slot = tick / AUX_EVERY;
    (slot + hash_rng(s, k, x, y) % slots) % slots == 0
}

/// `true` when `m` can hold water: a porous powder, or anything with an authored
/// [`Material::absorbency`] — which is how a *solid* like cardboard opts in.
///
/// Every rule in this module routes its "can this cell take water" question through here,
/// so widening it is all it takes to make a solid wettable. The powder half is still keyed
/// on `porosity` rather than `absorbency` so that no powder's behaviour moves.
#[inline]
fn absorbent(table: &MaterialTable, m: u8) -> bool {
    (table.class(m) == MaterialClass::Powder && table.porosity(m) > 0.0)
        || table.absorbency(m) > 0.0
}

/// Wetness a cell of `m` holds against gravity, i.e. its own field capacity.
///
/// [`FIELD_CAPACITY`] is a sand number — 19% of saturation, which is what a granular bed
/// holds once it has drained through. An absorbent material holds water *because* it is
/// absorbent, so its capacity is its `absorbency`, and only the floor of the two is
/// granular.
///
/// Getting this wrong is what made soaked cardboard hollow out from the far side while the
/// layer touching the water stayed perfectly intact. Draining every cell to 48 put all of
/// them below [`SOG_MIN_WETNESS`], so their wear froze and they could never fail; only the
/// bottom-most cell of a column, having nowhere to drain to, ever reached the threshold.
/// The board was behaving as a drain pipe rather than a sponge, which is also why the water
/// it was supposedly soaking up kept arriving at the bottom face.
#[inline]
fn field_capacity(table: &MaterialTable, m: u8) -> u8 {
    ((table.absorbency(m) * 255.0) as u8).max(FIELD_CAPACITY)
}

/// `true` when `m` is a solid that soaks — the set that goes soggy and eventually fails.
#[inline]
fn soggy_solid(table: &MaterialTable, m: u8) -> bool {
    table.class(m) == MaterialClass::Solid && table.absorbency(m) > 0.0
}

/// A liquid cell touching absorbent powder is consumed and its water pushed into the
/// contact cell, spilling into that cell's neighbours when it saturates.
///
/// Returns whether it absorbed, which is what keeps the chunk's wet fuse alive.
#[inline]
fn absorb_cell(
    l: &mut Layer,
    table: &MaterialTable,
    s: u64,
    tick: u64,
    x: u16,
    y: u16,
    i: usize,
) -> bool {
    if (tick + hash_rng(s, 0, x, y) % ABSORB_EVERY) % ABSORB_EVERY != 0 {
        return false;
    }
    // absorption consumes the liquid cell, so a cell carrying suspended solids is
    // off limits — otherwise the grains it is carrying are destroyed with it
    if l.susp_conc[i] > 0 {
        return false;
    }
    let mut contact: Option<(u16, u16)> = None;
    for (dx, dy) in [(0i32, 1i32), (-1, 0), (1, 0), (0, -1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if !l.in_bounds(nx, ny) {
            continue;
        }
        let (nx, ny) = (nx as u16, ny as u16);
        let ni = l.idx(nx, ny);
        if absorbent(table, l.mat[ni]) && l.wetness[ni] < 255 {
            contact = Some((nx, ny));
            break;
        }
    }
    let Some((cx, cy)) = contact else {
        return false;
    };

    // A whole liquid cell is worth ABSORB_UNITS of wetness, so absorbing into a patch
    // with less free capacity than that throws water away. Without this guard a
    // saturated bed keeps drinking cells to top up the single unit it lost to
    // evaporation, and a pool drains into nothing.
    if patch_capacity(l, table, cx, cy) < ABSORB_UNITS {
        return false;
    }

    // the water cell is consumed
    l.mat[i] = 0;
    l.flags[i] = 0;
    l.clear_aux(i);
    l.touch(x, y);
    inject_wetness(l, table, cx, cy, ABSORB_UNITS);
    true
}

/// A dry absorbent cell sitting against water drinks one clean neighbouring cell into its
/// own patch — the powder-side twin of [`absorb_cell`].
///
/// Same conservation contract: it consumes a whole liquid cell and injects exactly
/// [`ABSORB_UNITS`], so it cannot mint or lose water and a grain never soaks more than the
/// patch can hold. The differences are only which side drives it and which water it will
/// take:
///
/// * It runs on the **dry grain**, so a scatter of grains submerged in a pond wet from the
///   water around them rather than only where a pool drains onto them or a deposit lands on
///   top — the behaviour that made sand look dry until dust settled on it.
/// * It refuses **cloudy** water (`susp_conc > 0`), because consuming that cell would
///   destroy the solids it carries, and prefers a clean neighbour on any side. A lone grain
///   whose patch cannot hold a whole cell still refuses (its own capacity is one cell's
///   worth short), and is left to read as wet through the renderer's submerged cue instead.
#[inline]
fn soak_submerged(l: &mut Layer, table: &MaterialTable, x: u16, y: u16) -> bool {
    if patch_capacity(l, table, x, y) < ABSORB_UNITS {
        return false;
    }
    for (dx, dy) in [(0i32, 1i32), (-1, 0), (1, 0), (0, -1)] {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if !l.in_bounds(nx, ny) {
            continue;
        }
        let (nx, ny) = (nx as u16, ny as u16);
        let ni = l.idx(nx, ny);
        if table.class(l.mat[ni]) != MaterialClass::Liquid || l.susp_conc[ni] > 0 {
            continue;
        }
        // the clean liquid cell is consumed into this grain's patch, exactly as
        // `absorb_cell` consumes it from the other side
        l.mat[ni] = 0;
        l.flags[ni] = 0;
        l.clear_aux(ni);
        l.touch(nx, ny);
        inject_wetness(l, table, x, y, ABSORB_UNITS);
        return true;
    }
    false
}

/// How far absorption reaches to find room for a cell of water, as a Chebyshev radius.
///
/// It was 1 — the 8 neighbours — and that quietly made **thin absorbent structures
/// waterproof**, which is every cardboard box there will ever be. A cell of water is
/// [`ABSORB_UNITS`] and a one-cell-thick sheet only ever offers three cells of the patch, so
/// `patch_capacity` returned 765 against a required 896 and absorption refused for ever: a
/// cardboard shelf under standing water stayed bone dry for 3000 ticks.
///
/// 3 gives a 7x7 reach, so a sheet can find room along itself. Both the measure and the fill
/// work outward ring by ring, so a deep bed still fills its 3x3 and never looks further —
/// existing behaviour is unchanged, and only the cases that used to refuse outright now
/// reach past the first ring.
const ABSORB_REACH: i32 = 3;

/// Visits cells within [`ABSORB_REACH`] of `(x, y)`, centre first and then ring by ring, so
/// water lands as close to where it entered as it can. `f` returns `true` to stop.
#[inline]
fn for_patch(l: &Layer, x: u16, y: u16, mut f: impl FnMut(usize) -> bool) {
    for r in 0..=ABSORB_REACH {
        for dy in -r..=r {
            for dx in -r..=r {
                // ring r only — the interior belonged to a nearer pass
                if r > 0 && dx.abs() != r && dy.abs() != r {
                    continue;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if !l.in_bounds(nx, ny) {
                    continue;
                }
                if f(l.idx(nx as u16, ny as u16)) {
                    return;
                }
            }
        }
    }
}

/// Free wetness capacity of the absorbent cells in reach of the contact cell — the same
/// region, in the same order, that `inject_wetness` fills.
pub(crate) fn patch_capacity(l: &Layer, table: &MaterialTable, x: u16, y: u16) -> u32 {
    let mut cap = 0u32;
    for_patch(l, x, y, |ni| {
        if absorbent(table, l.mat[ni]) {
            cap += (255 - l.wetness[ni]) as u32;
        }
        false
    });
    cap
}

/// Greedy fill of `(x, y)` to 255, surplus spread over the 8 neighbours, remainder
/// abandoned. No allocation, no queue.
pub(crate) fn inject_wetness(l: &mut Layer, table: &MaterialTable, x: u16, y: u16, mut units: u32) {
    // Nothing here assumes the cells are absorbent, the centre least of all: a cell that has
    // just been emptied gets its soak paid out through this, and dumping wetness into a hole
    // would put water somewhere that cannot hold it and that nothing will ever dry.
    let mut fill: Vec<usize> = Vec::new();
    for_patch(l, x, y, |ni| {
        if absorbent(table, l.mat[ni]) && l.wetness[ni] < 255 {
            fill.push(ni);
        }
        false
    });
    let w = l.w as usize;
    for ni in fill {
        let take = ((255 - l.wetness[ni]) as u32).min(units);
        if take == 0 {
            continue;
        }
        l.wetness[ni] += take as u8;
        units -= take;
        let (nx, ny) = ((ni % w) as u16, (ni / w) as u16);
        l.touch(nx, ny);
        l.mark_damp(nx, ny);
        if units == 0 {
            return;
        }
    }
}

/// Water above field capacity falls into the absorbent cell below.
///
/// No gradient test and no dead-band: gravity does not need a difference in wetness to
/// act, only somewhere to go. That is the whole difference between this and
/// [`capillary_cell`], and the reason a pool now drains into a deep bed instead of
/// stalling on a shallow damp patch.
///
/// Only the receiver's *free capacity* limits the transfer, so a saturated column stops
/// draining and the bed fills from the bottom up — which is what a perched water table
/// looks like, and what makes a saturated pile shed drips at its free faces.
#[inline]
fn drain_cell(l: &mut Layer, table: &MaterialTable, x: u16, y: u16, i: usize) -> bool {
    let fc = field_capacity(table, l.mat[i]);
    if l.wetness[i] <= fc {
        return false;
    }
    let ny = y as i32 + 1;
    if !l.in_bounds(x as i32, ny) {
        return false;
    }
    let ny = ny as u16;
    let ni = l.idx(x, ny);
    if !absorbent(table, l.mat[ni]) {
        return false;
    }
    let t = (l.wetness[i] - fc)
        .min(DRAIN_MAX_TRANSFER)
        .min(255 - l.wetness[ni]);
    if t == 0 {
        return false;
    }
    let was_dry = l.wetness[ni] == 0;
    l.wetness[i] -= t;
    l.wetness[ni] += t;
    // Wetting a dry cell changes its repose and density, so the powder solver has to look
    // at it again; a cell that was already wet only needs a re-bake, and waking it would
    // hand back the sleep win this pass exists to keep.
    if was_dry {
        l.touch(x, ny);
    } else {
        l.touch_render(x, ny);
    }
    l.touch_render(x, y);
    l.mark_damp(x, ny);
    true
}

/// One weighted neighbour per tick; upward is chosen 1.5x as often as any other
/// direction, which is what makes the front climb.
#[inline]
fn capillary_cell(l: &mut Layer, table: &MaterialTable, x: u16, y: u16, i: usize) -> bool {
    let (dx, dy) = match l.rng.below(9) {
        0..=2 => (0i32, -1i32),
        3..=4 => (0, 1),
        5..=6 => (-1, 0),
        _ => (1, 0),
    };
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    if !l.in_bounds(nx, ny) {
        return false;
    }
    let (nx, ny) = (nx as u16, ny as u16);
    let ni = l.idx(nx, ny);
    if !absorbent(table, l.mat[ni]) {
        return false;
    }
    let delta = l.wetness[i] as i32 - l.wetness[ni] as i32;
    if delta <= CAPILLARY_MIN_DELTA {
        return false;
    }
    let t = ((delta / 4) as u8)
        .min(CAPILLARY_MAX_TRANSFER)
        .min(l.wetness[i])
        .min(255 - l.wetness[ni]);
    if t == 0 {
        return false;
    }
    l.wetness[i] -= t;
    l.wetness[ni] += t;
    l.touch(x, y);
    l.touch(nx, ny);
    l.mark_damp(nx, ny);
    true
}

/// A soaked solid loses its stiffness. Nothing is destroyed — see [`sag_cell`], which is
/// what the lost stiffness actually does.
///
/// Wear is monotonic and survives drying: a box that has been soaked once is weakened for
/// good, which is both true of corrugated board and the more interesting rule — it means a
/// leak you mopped up has still cost you something. It accumulates only above
/// [`SOG_MIN_WETNESS`], so a merely damp box lasts forever.
#[inline]
fn soggy_cell(l: &mut Layer, table: &MaterialTable, s: u64, tick: u64, x: u16, y: u16, i: usize) {
    let m = l.mat[i];
    if !soggy_solid(table, m) || l.wetness[i] < SOG_MIN_WETNESS || l.wear[i] == u8::MAX {
        return;
    }
    if !due(s, 3, tick, x, y, SOG_EVERY) {
        return;
    }
    let step = ((SOG_WEAR_STEP as f32 * table.absorbency(m)).round() as u8).max(1);
    l.wear[i] = l.wear[i].saturating_add(step);
    // a re-bake, not a reason to restart the solvers — the sag rule wakes what it moves
    l.touch_render(x, y);
    l.mark_damp(x, y);
}

/// A weakened cell with nothing under it droops into the gap, **displacing** rather than
/// disappearing.
///
/// The first cut of this deleted the cell at full wear, which is wrong twice over: it loses
/// material, and a box that vanishes pixel by pixel does not read as a wet box — it reads
/// as a box being eaten. Cardboard that has gone soft does not evaporate, it sags, and
/// every cell that sags is still there afterwards. The move is a swap, so material is
/// conserved exactly and there is no water to pay out.
///
/// # What makes it a sag rather than a collapse
///
/// A cell is **pinned** if any of its four neighbours is a solid that is still sound —
/// wear under [`SAG_MIN_WEAR`]. Sound board holds soft board up, so the ends of a wetted
/// floor stay hung off the dry walls they join while the middle droops, which is a curve
/// rather than a slab falling out of the bottom. It also makes the deformation spread the
/// way the wetting does: the anchors are wherever the board is still dry, so the sag
/// deepens as the soak creeps outward, and a box wetted only in one corner sags only there.
///
/// Only empty space below counts as somewhere to go. Sagging into a liquid would need a
/// story about buoyancy — cardboard is 300 kg/m³ and floats — and the case that matters is
/// a wetted floor over air.
///
/// Rate-limited to one cell per [`SAG_EVERY`] ticks, hash-phased per cell like every other
/// rule here. Unlimited, a soft sheet would descend at a cell a tick, which is falling.
#[inline]
fn sag_cell(
    l: &mut Layer,
    table: &MaterialTable,
    s: u64,
    tick: u64,
    x: u16,
    y: u16,
    i: usize,
) -> bool {
    if !soggy_solid(table, l.mat[i]) || l.wear[i] < SAG_MIN_WEAR {
        return false;
    }
    // a cell the powder or liquid pass already moved is retired for this tick
    if l.flags[i] & FLAG_MOVED != 0 {
        return false;
    }
    if !due(s, 4, tick, x, y, SAG_EVERY) {
        return false;
    }
    if y + 1 >= l.h || !table.is_empty(l.mat_at(x, y + 1)) {
        return false;
    }
    for (dx, dy) in [(0i32, -1i32), (-1, 0), (1, 0), (0, 1)] {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if !l.in_bounds(nx, ny) {
            continue;
        }
        let ni = l.idx(nx as u16, ny as u16);
        if table.is_solid(l.mat[ni]) && l.wear[ni] < SAG_MIN_WEAR {
            return false; // pinned to sound board
        }
    }
    if !slope_ok(l, table, x, y) {
        return false;
    }
    l.commit_move((x, y), (x, y + 1));
    true
}

/// May the cell at `(x, y)` drop a row without pulling away from the material beside it?
///
/// This is the difference between a sheet sagging and a sheet **disintegrating**. Two
/// earlier rules both failed, and how they failed is the argument for this one:
///
/// * With no cohesion at all, every soft cell fell on its own hash phase and a wetted floor
///   came apart into a descending cloud of loose pixels.
/// * Requiring only that a cell still *touch* something afterwards was no better. A flat
///   sheet satisfies it trivially — each cell lands beside the one that went before — so the
///   whole sheet marched downwards a row at a time, still nominally connected but ragged,
///   and 20 rows below where it started.
///
/// Bulk descent is the thing to forbid, so the test is a **slope limit**: a cell may not end
/// up more than one row below the columns either side of it, which means each neighbour must
/// hold material at this cell's own row or the one below. Out of bounds counts as immovable.
///
/// Combined with the pin to sound board, that bounds the whole deformation. Depth can only
/// increase by one per column away from an anchor, so a span pinned at both ends settles
/// into a shallow V no deeper than half its width and then **stops** — a sagging floor,
/// which is what a wet one does, rather than a floor that leaves.
#[inline]
fn slope_ok(l: &Layer, table: &MaterialTable, x: u16, y: u16) -> bool {
    for dx in [-1i32, 1] {
        let nx = x as i32 + dx;
        if !l.in_bounds(nx, y as i32) {
            return false;
        }
        let level = table.is_solid(l.mat[l.idx(nx as u16, y)]);
        let below = y + 1 < l.h && table.is_solid(l.mat[l.idx(nx as u16, y + 1)]);
        if !(level || below) {
            return false;
        }
    }
    true
}

/// An exposed wet cell dries slowly; the last unit leaves a stain.
#[inline]
fn evaporate_cell(
    l: &mut Layer,
    table: &MaterialTable,
    s: u64,
    tick: u64,
    x: u16,
    y: u16,
    i: usize,
) {
    if !due(s, 1, tick, x, y, EVAPORATE_EVERY) {
        return;
    }
    let mut exposed = false;
    for (dx, dy) in [(0i32, -1i32), (-1, 0), (1, 0), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if !l.in_bounds(nx, ny) {
            continue;
        }
        if table.is_empty(l.mat_at(nx as u16, ny as u16)) {
            exposed = true;
            break;
        }
    }
    if !exposed {
        return;
    }
    l.wetness[i] -= 1;
    if l.wetness[i] == 0 {
        l.dirt[i] = l.dirt[i].saturating_add(STAIN_STEP);
        // the dry-out is a real state change (repose and density both jump, and the
        // stain appears), so this one is worth a wake
        l.touch(x, y);
    } else {
        // one unit of wetness is a re-bake, not a reason to restart the solvers
        l.touch_render(x, y);
        l.mark_damp(x, y);
    }
}

/// Faces a drop can leave through, in preference order. Straight down is the intuition,
/// but it is *unreachable* for a cell at rest: `step_powder` runs before the wet rules,
/// so a powder cell with an empty cell under it has already fallen into it and a resting
/// pile only ever has free faces on its slope. Testing only `(0, 1)` therefore meant no
/// absorbed water ever came back to the world and the absorption mass loop never closed.
const DRIP_FACES: [(i32, i32); 5] = [(0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)];

/// Total wetness held by the 3x3 patch around `(x, y)` — the same patch
/// `inject_wetness` fills and `patch_capacity` measures, and the pot a drop is paid out
/// of.
fn patch_wetness(l: &Layer, table: &MaterialTable, x: u16, y: u16) -> u32 {
    let mut sum = 0u32;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if !l.in_bounds(nx, ny) {
                continue;
            }
            let ni = l.idx(nx as u16, ny as u16);
            if absorbent(table, l.mat[ni]) {
                sum += l.wetness[ni] as u32;
            }
        }
    }
    sum
}

/// A saturated pile with a free face sheds a water cell, closing the mass loop that
/// absorption opened. A cell of water is worth [`ABSORB_UNITS`] of wetness and the drop
/// is paid in full out of the local patch *before* it exists — a partial payment used to
/// mint 71% of the drop out of nothing.
#[inline]
fn drip_cell(l: &mut Layer, table: &MaterialTable, s: u64, tick: u64, x: u16, y: u16, i: usize) {
    if l.wetness[i] <= DRIP_MIN_WETNESS {
        return;
    }
    if !due(s, 2, tick, x, y, DRIP_EVERY) {
        return;
    }
    let liq = table.wetting_liquid();
    if liq == 0 {
        return;
    }
    // affordability first: no partial drops, because a cell of water is indivisible
    if patch_wetness(l, table, x, y) < ABSORB_UNITS {
        return;
    }
    let mut target: Option<(u16, u16)> = None;
    for (dx, dy) in DRIP_FACES {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if !l.in_bounds(nx, ny) {
            continue;
        }
        if table.is_empty(l.mat_at(nx as u16, ny as u16)) {
            target = Some((nx as u16, ny as u16));
            break;
        }
    }
    let Some((tx, ty)) = target else { return };

    let mut owed = ABSORB_UNITS;
    let take = (l.wetness[i] as u32).min(owed);
    l.wetness[i] -= take as u8;
    owed -= take;
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if owed == 0 {
                break;
            }
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if !l.in_bounds(nx, ny) {
                continue;
            }
            let (nx, ny) = (nx as u16, ny as u16);
            let ni = l.idx(nx, ny);
            if !absorbent(table, l.mat[ni]) {
                continue;
            }
            let take = (l.wetness[ni] as u32).min(owed);
            l.wetness[ni] -= take as u8;
            owed -= take;
            if take > 0 {
                l.touch(nx, ny);
            }
        }
    }
    debug_assert_eq!(owed, 0, "patch_wetness over-reported the pot");

    l.set_mat(tx, ty, liq);
    l.touch(x, y);
}
