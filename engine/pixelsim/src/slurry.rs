//! Suspended-solids transport: Stokes settle, deposit, entrainment, re-suspension.
//! Single species per cell; `susp_mat`/`susp_conc` stay adjacent so widening to 4
//! slots later is mechanical.

use crate::cell::FLAG_MOVED;
use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::step::for_awake_cells_bottom_up;

/// scaled so a 500 µm grain of quartz in water settles at 0.5 cells/tick; sand at
/// 400 µm therefore comes out near 0.32 and gold_dust at 60 µm near 0.08
pub const SETTLE_C: f32 = 0.5 / 1650.0;
/// one entrained grain saturates one cell's suspension channel, so total
/// concentration is conserved by transfers and a deposit hands exactly one grain back
pub const ENTRAIN_UNITS: u8 = 255;
/// `true` when a suspension channel is holding a whole grain, i.e. as much as one
/// cell can carry. Spelled as equality rather than `>= ENTRAIN_UNITS` because
/// `ENTRAIN_UNITS` saturates the `u8`, which makes the ordering comparison vacuous.
#[inline]
pub fn is_full_grain(conc: u8) -> bool {
    conc == ENTRAIN_UNITS
}

/// grains finer than `RESUSPEND_K * flow_mag` µm get picked back up. Scaled so that
/// gold_dust (60 µm) resuspends in briskly moving water and sand (400 µm) never does,
/// given that `flow_mag` saturates at 255.
pub const RESUSPEND_K: f32 = 1.0;

#[inline]
pub fn v_settle(table: &MaterialTable, susp: u8, fluid: u8) -> f32 {
    let d = table.grain(susp) / 500.0;
    (SETTLE_C * (table.density_grain_or_bulk(susp) - table.density(fluid)) * d * d
        / table.viscosity(fluid))
    .clamp(0.0, 1.0)
}

/// Can the liquid cell at `into_i` take another grain of `m`? Single species per
/// cell, so a mismatched species is refused and the caller falls back to displacing.
///
/// A grain is indivisible in this representation: entrainment adds [`ENTRAIN_UNITS`] to
/// a `u8`, so the cell must have that much *headroom*, not merely some. Admitting a
/// partly loaded cell and then relying on `saturating_add` silently discarded the
/// overflow — 17.7% of all powder mass in a mixed scene over 1200 ticks. The honest
/// alternative would be a sub-grain representation the `u8` channel cannot express, so
/// the caller displaces the liquid instead, which conserves both.
#[inline]
pub fn can_entrain(l: &Layer, m: u8, into_i: usize) -> bool {
    l.susp_conc[into_i] == 0
        || (l.susp_mat[into_i] == m && l.susp_conc[into_i].checked_add(ENTRAIN_UNITS).is_some())
}

/// Consumes the powder cell at `from` into the suspension channel of the liquid cell
/// at `into`. The liquid stays put; the vacated cell becomes empty.
///
/// The grain's **pore water travels with it**, in the carrier's own `wetness` channel —
/// see [`carried_film`]. Without that, `clear_aux` on the vacated cell simply deleted it:
/// up to 255 units per grain, on every lap of the entrain/deposit cycle, which is a second
/// unbounded water sink behind the same symptom [`deposit`] documents. It is slower than
/// the cell-per-lap one and so survived the first fix — 305 of 1790 water cells over 1600
/// ticks, still falling linearly at 0.25 cells/tick with no equilibrium in sight.
pub fn entrain(layer: &mut Layer, table: &MaterialTable, from: (u16, u16), into: (u16, u16)) {
    let fi = layer.idx(from.0, from.1);
    let ii = layer.idx(into.0, into.1);
    let m = layer.mat[fi];
    if !can_entrain(layer, m, ii) {
        return;
    }
    let _ = table;
    layer.susp_mat[ii] = m;
    // `can_entrain` guarantees the headroom, so this cannot clip
    layer.susp_conc[ii] += ENTRAIN_UNITS;
    // one grain per cell (a second `+= 255` cannot fit in the `u8`), so the film cannot
    // exceed one cell's saturation and this addition cannot clip either
    layer.wetness[ii] = layer.wetness[ii].saturating_add(layer.wetness[fi]);
    layer.mat[fi] = 0;
    layer.flags[fi] = 0;
    layer.clear_aux(fi);
    layer.touch(from.0, from.1);
    layer.touch(into.0, into.1);
    layer.stats.moves += 1;
}

/// Turns a saturated suspension back into a powder cell, **re-homing the liquid that
/// carried it** — see [`rehome_carrier`]. Returns whether the grain was placed; a grain
/// whose carrier has nowhere to go stays in suspension and tries again next tick.
///
/// Refuses anything short of a full grain ([`ENTRAIN_UNITS`]). One grain in, one grain
/// out is the whole basis of the concentration bookkeeping; minting a whole cell of
/// powder from the 10 units a species-conflict call happened to be holding amplified
/// mass 17x in a single tick.
///
/// # Why the carrier cannot simply be overwritten
///
/// It used to be, documented as "the accepted approximation: the visible outcome is a
/// wet deposit, not an audited water budget". The approximation is not one-off, because
/// [`resuspend_cell`] closes it into a **cycle**: entrainment frees the powder's cell and
/// the grain rides the water as a volumeless passenger, deposit then spends a water cell
/// to give the grain a cell back, and the deposited grain — being fine enough to have
/// been picked up in the first place — is immediately eligible to be picked up again. Each
/// lap costs exactly one cell of water and conserves the powder, so a handful of fine
/// grains is an *unbounded* water sink rather than a small error. Measured on 63 cells of
/// silt under a 1790-cell pool: 99% of the water gone in 400 ticks, silt count unchanged.
/// 18 cells of silt ate 3664. Sand (400 µm) is untouched by any of this and lost nothing,
/// which is what made the bug look like a property of fine powders "absorbing" too much.
pub fn deposit(layer: &mut Layer, table: &MaterialTable, at: (u16, u16)) -> bool {
    let i = layer.idx(at.0, at.1);
    let m = layer.susp_mat[i];
    if m == 0 || table.class(m) != MaterialClass::Powder {
        return false;
    }
    if !is_full_grain(layer.susp_conc[i]) {
        return false;
    }
    let carrier = layer.mat[i];
    let film = carried_film(layer, i);
    let Some(route) = rehome_carrier(layer, table, at.0, at.1, carrier, film) else {
        return false;
    };
    layer.mat[i] = m;
    layer.susp_mat[i] = 0;
    layer.susp_conc[i] = 0;
    layer.head[i] = 0;
    // The grain gets its own pore water back and nothing more. Assigning a flat 255 here
    // (the old rule) mints wetness from nothing; the carrier is a whole cell of liquid that
    // has just been re-homed, so claiming its water as well would count it twice. A grain
    // that entrained dry deposits dry, sitting in the pool that closed over it, and
    // `absorb_cell` wets it within `ABSORB_EVERY` ticks under the usual capacity guard.
    layer.wetness[i] = film;
    layer.touch(at.0, at.1);
    layer.mark_damp(at.0, at.1);
    if route == Carrier::SoakedAway {
        // the film is already on the cell, so only the carrier itself is owed
        crate::wet::inject_wetness(layer, table, at.0, at.1, crate::wet::ABSORB_UNITS);
    }
    true
}

/// The pore water a suspended grain is carrying in its liquid cell, in wetness units.
///
/// Liquids have no use for the `wetness` channel of their own — absorption reads it only
/// on powders — so it is where [`entrain`] parks the film it lifted off the grain and where
/// [`deposit`] finds it again. Reads as zero for an unladen cell, so this is safe on any
/// liquid.
#[inline]
fn carried_film(l: &Layer, i: usize) -> u8 {
    l.wetness[i]
}

/// Moves a share of the film along with a transfer of suspension, so the pore water stays
/// with the grain it came off. Call *after* updating `susp_conc[from]`; `conc` is what that
/// cell held before the transfer of `amount`.
///
/// The invariant this exists to keep is **film == 0 whenever susp_conc == 0**. A film left
/// behind on a cell whose grain has settled onward is not merely misplaced: that cell is
/// now an ordinary unladen liquid, so `absorb_cell` will happily consume it, and
/// `clear_aux` deletes the film on the way out. Hence the exact drain of the remainder
/// once the concentration hits zero rather than leaving truncation dust behind.
#[inline]
fn transfer_film(l: &mut Layer, from: usize, to: usize, amount: u8, conc: u8) {
    let film = l.wetness[from];
    if film == 0 || conc == 0 {
        return;
    }
    let t = if l.susp_conc[from] == 0 {
        film
    } else {
        ((film as u32 * amount as u32) / conc as u32) as u8
    };
    l.wetness[from] -= t;
    l.wetness[to] = l.wetness[to].saturating_add(t);
}

/// What happened to the carrier liquid, and therefore what the caller still owes.
#[derive(PartialEq, Eq)]
enum Carrier {
    /// moved out as a whole cell of liquid; the books are square
    Moved,
    /// spent into the surrounding powder; the caller owes the injection
    SoakedAway,
}

/// Finds somewhere for the cell of `carrier` liquid that a depositing grain is about to
/// evict. `None` means nowhere, and the caller must not deposit.
///
/// `Some(true)` means the carrier was spent as wetness and the caller still owes the
/// injection; `Some(false)` means it was moved and the books are already square.
///
/// Three routes, in order:
///
/// 1. **An adjacent empty cell**, preferring up. A grain settling near the waterline
///    displaces water into the air above it.
/// 2. **The free surface of its own column.** Drop a grain into a glass and the level
///    rises: the displaced water is not local to the grain, it joins the top of the
///    column. So walk up through the liquid to the first empty cell. Bounded by the
///    column, and it terminates at the surface in the case that matters — a bed under a
///    pool, which is where essentially every deposit happens.
/// 3. **Soaked into the bed**, if the 3x3 patch has a full [`ABSORB_UNITS`] of free
///    wetness capacity. This is the same trade absorption makes, so it is bounded by the
///    same guard and cannot leak: a sealed pocket of slurry against dry sand settles out,
///    and once the sand is saturated route 3 closes.
///
/// If all three fail the vessel is water-tight and brim-full, and an incompressible
/// column genuinely has no room for the grain to sink through. Staying suspended is the
/// honest answer, and it is self-clearing — any evaporation, drainage or leak reopens
/// route 1 or 3.
fn rehome_carrier(
    l: &mut Layer,
    table: &MaterialTable,
    x: u16,
    y: u16,
    carrier: u8,
    film: u8,
) -> Option<Carrier> {
    for (dx, dy) in [(0i32, -1i32), (-1, 0), (1, 0), (0, 1)] {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if !l.in_bounds(nx, ny) {
            continue;
        }
        let (nx, ny) = (nx as u16, ny as u16);
        if table.is_empty(l.mat_at(nx, ny)) {
            l.set_mat(nx, ny, carrier);
            return Some(Carrier::Moved);
        }
    }
    let mut yy = y;
    while yy > 0 {
        yy -= 1;
        let m = l.mat_at(x, yy);
        if table.is_empty(m) {
            l.set_mat(x, yy, carrier);
            return Some(Carrier::Moved);
        }
        if table.class(m) != MaterialClass::Liquid {
            break;
        }
    }
    // The patch is measured while this cell is still liquid, so it counts as zero free
    // capacity — but the grain about to land on it arrives dry, and `inject_wetness`
    // starts there. Add that 255 back or route 3 demands the whole carrier from the
    // neighbours alone and refuses deposits it could well afford.
    let own = if table.porosity(l.susp_mat[l.idx(x, y)]) > 0.0 {
        255 - film as u32
    } else {
        0
    };
    if crate::wet::patch_capacity(l, table, x, y) + own >= crate::wet::ABSORB_UNITS {
        return Some(Carrier::SoakedAway);
    }
    None
}

pub fn step_slurry(layer: &mut Layer, table: &MaterialTable, tick: u64) {
    let _ = tick;
    for_awake_cells_bottom_up(layer, |l, x, y| {
        let i = l.idx(x, y);
        let m = l.mat[i];
        match table.class(m) {
            MaterialClass::Liquid => settle_cell(l, table, x, y, i, m),
            MaterialClass::Powder => resuspend_cell(l, table, x, y, i, m),
            _ => {}
        }
    });
}

#[inline]
fn settle_cell(l: &mut Layer, table: &MaterialTable, x: u16, y: u16, i: usize, fluid: u8) {
    let susp = l.susp_mat[i];
    let conc = l.susp_conc[i];
    if susp == 0 || conc == 0 {
        return;
    }
    let v = v_settle(table, susp, fluid);
    if v <= 0.0 {
        return;
    }
    // at least one unit while any is left: rounding `v * conc` down to zero strands
    // the last few units of every cell and the suspension never reaches the bed
    let want = ((v * conc as f32).round().clamp(0.0, 255.0) as u8).clamp(1, conc);

    // nowhere below to sink into — this cell is the bed
    if y + 1 >= l.h {
        settle_at_bed(l, table, x, y, i, susp, conc);
        return;
    }
    let bi = l.idx(x, y + 1);
    let bm = l.mat[bi];
    if table.class(bm) != MaterialClass::Liquid {
        settle_at_bed(l, table, x, y, i, susp, conc);
        return;
    }
    // Species conflict. One species per cell, so the cell below cannot accept this one:
    // it is a floor as far as this cell is concerned, and the at-bed rule (concentrate,
    // then bed-load creep, then drop out at a full grain) is exactly the right
    // behaviour. Nothing here may *deposit* a partial load — the old rule handed the
    // faster settler a whole grain of powder for whatever fraction it held, which is
    // where the 17x mass amplification came from. The cell below still deposits if it
    // has genuinely filled up, which turns the floor into a real one.
    if l.susp_conc[bi] > 0 && l.susp_mat[bi] != susp {
        if is_full_grain(l.susp_conc[bi]) {
            deposit(l, table, (x, y + 1));
        }
        settle_at_bed(l, table, x, y, i, susp, conc);
        return;
    }

    let room = 255 - l.susp_conc[bi];
    let amount = want.min(room);
    if amount == 0 {
        // the cell below is already saturated, so this one is effectively at the bed
        settle_at_bed(l, table, x, y, i, susp, conc);
        return;
    }
    l.susp_mat[bi] = susp;
    l.susp_conc[bi] += amount;
    l.susp_conc[i] -= amount;
    if l.susp_conc[i] == 0 {
        l.susp_mat[i] = 0;
    }
    transfer_film(l, i, bi, amount, conc);
    l.touch(x, y);
    l.touch(x, y + 1);
    if is_full_grain(l.susp_conc[bi]) {
        let below_is_liquid =
            y + 2 < l.h && table.class(l.mat_at(x, y + 2)) == MaterialClass::Liquid;
        if !below_is_liquid {
            deposit(l, table, (x, y + 1));
        }
    }
}

/// A bed cell cannot settle any further, so it concentrates instead: it drops out as a
/// grain once it holds a full cell's worth, otherwise it pulls suspension in from the
/// neighbouring bed cells (bed-load creep).
///
/// Without this, laterally moving water dilutes a few grains over dozens of bed cells,
/// none of them ever reaches a full grain, and the suspension is stranded forever. The
/// pull is bounded by the receiving cell's headroom, so total concentration — and
/// therefore mass — is conserved, and the largest concentration in a bed segment only
/// ever increases, so the process terminates.
#[inline]
fn settle_at_bed(
    l: &mut Layer,
    table: &MaterialTable,
    x: u16,
    y: u16,
    i: usize,
    susp: u8,
    conc: u8,
) {
    if is_full_grain(conc) {
        deposit(l, table, (x, y));
        return;
    }
    // Push downstream, in the same direction the scan is running, so a whole run of
    // dilute bed cells consolidates in one pass. Only into a neighbour that already
    // carries the same species: merging with existing sediment is bed-load creep,
    // sweeping across clean bed is not.
    let dir: i32 = if l.tick & 1 == 0 { 1 } else { -1 };
    let nx = x as i32 + dir;
    if nx < 0 || nx >= l.w as i32 {
        return;
    }
    let nx = nx as u16;
    let ni = l.idx(nx, y);
    if table.class(l.mat[ni]) != MaterialClass::Liquid
        || l.susp_conc[ni] == 0
        || l.susp_mat[ni] != susp
    {
        return;
    }
    // the neighbour must be bed too, or the bed would push sediment back up a column
    if y + 1 < l.h && table.class(l.mat_at(nx, y + 1)) == MaterialClass::Liquid {
        return;
    }
    let give = conc.min(255 - l.susp_conc[ni]);
    if give == 0 {
        return;
    }
    l.susp_conc[ni] += give;
    l.susp_conc[i] -= give;
    if l.susp_conc[i] == 0 {
        l.susp_mat[i] = 0;
    }
    transfer_film(l, i, ni, give, conc);
    l.touch(x, y);
    l.touch(nx, y);
    if is_full_grain(l.susp_conc[ni]) {
        deposit(l, table, (nx, y));
    }
}

/// A settled grain fine enough for the local flow gets picked back up into an
/// adjacent liquid cell.
#[inline]
fn resuspend_cell(l: &mut Layer, table: &MaterialTable, x: u16, y: u16, i: usize, m: u8) {
    // a grain that already moved this tick is done: without this a grain could fall in
    // the powder pass and then be picked up again by the slurry pass in the same tick
    if l.flags[i] & FLAG_MOVED != 0 {
        return;
    }
    let flow = l.coarse.flow_at(x, y);
    if flow == 0 {
        return;
    }
    if table.grain(m) >= RESUSPEND_K * flow as f32 {
        return;
    }
    let mut sides = [(0u16, 0u16); 4];
    let mut n = 0usize;
    for (dx, dy) in [(0i32, -1i32), (-1, 0), (1, 0), (0, 1)] {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if !l.in_bounds(nx, ny) {
            continue;
        }
        let (nx, ny) = (nx as u16, ny as u16);
        let ni = l.idx(nx, ny);
        if table.class(l.mat[ni]) == MaterialClass::Liquid && can_entrain(l, m, ni) {
            sides[n] = (nx, ny);
            n += 1;
        }
    }
    if n == 0 {
        return;
    }
    let (nx, ny) = sides[l.rng.below(n as u32) as usize];
    entrain(l, table, (x, y), (nx, ny));
}
