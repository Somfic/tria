//! Adversarial-review reproductions. Each of these once failed and documented a real
//! defect; all nine are fixed and every test here now runs as part of the normal suite,
//! so a regression on any of them breaks the build rather than hiding behind `--ignored`.
//!
//! What each one guards: A/B/D mass conservation across entrainment and deposition,
//! C the drip mass budget, E the one-move-per-tick rule for displaced cells,
//! F that the slow aux rules survive chunk sleep, G bed-load direction symmetry,
//! H chunk arithmetic on a layer taller than `u16::MAX - CHUNK_PX`.
use pixelsim::*;

fn t() -> MaterialTable {
    MaterialTable::embedded()
}
fn l() -> ReposeLut {
    ReposeLut::analytic()
}
fn id(t: &MaterialTable, n: &str) -> u8 {
    t.id(n).unwrap().0
}

/// total "grain-equivalents": powder cells + suspension/255
fn powder_mass(layer: &Layer, tab: &MaterialTable) -> f64 {
    let mut m = 0.0;
    for i in 0..layer.mat.len() {
        if tab.class(layer.mat[i]) == MaterialClass::Powder {
            m += 1.0;
        }
        if layer.susp_conc[i] > 0 {
            m += layer.susp_conc[i] as f64 / 255.0;
        }
    }
    m
}
fn put(l: &mut Layer, x: u16, y: u16, m: u8) {
    let i = l.idx(x, y);
    l.mat[i] = m;
}
fn count(layer: &Layer, m: u8) -> usize {
    layer.mat.iter().filter(|&&x| x == m).count()
}

// ---------- A: entrain into a partially loaded cell destroys mass ----------
#[test]
fn a_entrain_into_partly_loaded_cell_loses_mass() {
    let tab = t();
    let water = id(&tab, "water");
    let gold = id(&tab, "gold_dust");
    let mut layer = Layer::new(8, 8, LayerSlot::Plant, 1);
    // one water cell already carrying 200/255 of gold_dust, one gold grain above it
    let wi = layer.idx(4, 4);
    layer.mat[wi] = water;
    layer.susp_mat[wi] = gold;
    layer.susp_conc[wi] = 200;
    layer.set_mat(4, 3, gold);

    let before = powder_mass(&layer, &tab);
    let mut layers = vec![layer];
    step_all_serial(&mut layers, &[], &tab, &l(), 1);
    let after = powder_mass(&layers[0], &tab);
    println!("A before={before} after={after}");
    assert!(
        (before - after).abs() < 1e-9,
        "mass changed: {before} -> {after}"
    );
}

// ---------- B: deposit from a partial suspension creates a whole grain ----------
#[test]
fn b_species_conflict_deposit_creates_mass() {
    let tab = t();
    let water = id(&tab, "water");
    let gold = id(&tab, "gold_dust"); // 60 um, fast settle
    let silt = id(&tab, "silt"); // 30 um, slow settle
    let mut layer = Layer::new(8, 8, LayerSlot::Plant, 1);
    for y in 3..7u16 {
        put(&mut layer, 4, y, water);
    }
    let a = layer.idx(4, 4);
    let b = layer.idx(4, 5);
    layer.susp_mat[a] = gold;
    layer.susp_conc[a] = 10; // faster species, tiny amount
    layer.susp_mat[b] = silt;
    layer.susp_conc[b] = 5; // slower species below
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }

    let before = powder_mass(&layer, &tab);
    let mut layers = vec![layer];
    step_all_serial(&mut layers, &[], &tab, &l(), 1);
    let after = powder_mass(&layers[0], &tab);
    println!(
        "B before={before} after={after} deposited={}",
        count(&layers[0], silt) + count(&layers[0], gold)
    );
    assert!(
        (before - after).abs() < 1e-9,
        "mass changed: {before} -> {after}"
    );
}

// ---------- C: drip emits a full water cell it could not pay for ----------
#[test]
fn c_drip_creates_water_it_cannot_pay_for() {
    let tab = t();
    let water = id(&tab, "water");
    let sand = id(&tab, "sand");
    let mut layer = Layer::new(8, 8, LayerSlot::Plant, 1);
    // ONE saturated sand cell, no absorbent neighbours, air below
    put(&mut layer, 4, 3, sand);
    let i = layer.idx(4, 3);
    layer.wetness[i] = 255;
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }

    let wet_before: u32 = layer.wetness.iter().map(|&w| w as u32).sum();
    let mut emitted = 0usize;
    for tick in 1..=600u64 {
        layer.tick = tick;
        layer.chunks.begin_tick();
        pixelsim::drip(&mut layer, &tab, tick);
        layer.chunks.end_tick();
        emitted = count(&layer, water);
        if emitted > 0 {
            println!("C drip at tick {tick}");
            break;
        }
    }
    let wet_after: u32 = layer.wetness.iter().map(|&w| w as u32).sum();
    println!(
        "C wetness paid = {} of the {} units a cell of water is worth; water cells = {emitted}",
        wet_before - wet_after,
        896
    );
    assert_eq!(
        emitted,
        0,
        "a drop was emitted for {} of {} units",
        wet_before - wet_after,
        896
    );
}

// ---------- F: a settled wet pile sleeps and then never evaporates/dries ----------
#[test]
fn f_sleeping_chunk_freezes_evaporation_and_drying() {
    let tab = t();
    let sand = id(&tab, "sand");
    let stone = id(&tab, "stone");
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 5);
    for x in 0..64u16 {
        put(&mut layer, x, 40, stone);
    }
    for x in 20..30u16 {
        put(&mut layer, x, 39, sand);
    }
    for x in 20..30u16 {
        let i = layer.idx(x, 39);
        layer.wetness[i] = 255;
    }
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }
    let mut layers = vec![layer];
    // EVAPORATE_EVERY is 40 ticks; give it 100x that
    for tick in 1..=4000u64 {
        step_all_serial(&mut layers, &[], &tab, &l(), tick);
    }
    let wet: u32 = layers[0].wetness.iter().map(|&w| w as u32).sum();
    let dirt: u32 = layers[0].dirt.iter().map(|&d| d as u32).sum();
    println!(
        "F after 4000 ticks: total wetness = {wet} (started 2550), stain = {dirt}, awake chunks = {}",
        layers[0].chunks.awake_count()
    );
    // 10 cells x ~100 evaporation windows in 4000 ticks: expect hundreds of units gone
    assert!(
        wet < 2000,
        "exposed wet sand stopped drying: only {} of 2550 units left in 4000 ticks",
        2550 - wet
    );
}

// ---------- D: whole-scene powder conservation ----------
#[test]
fn d_scene_conserves_powder_mass() {
    let tab = t();
    let (sand, silt, gold, water, stone) = (
        id(&tab, "sand"),
        id(&tab, "silt"),
        id(&tab, "gold_dust"),
        id(&tab, "water"),
        id(&tab, "stone"),
    );
    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 0xABCD);
    // sealed box
    for x in 0..128u16 {
        put(&mut layer, x, 127, stone);
        put(&mut layer, x, 0, stone);
    }
    for y in 0..128u16 {
        put(&mut layer, 0, y, stone);
        put(&mut layer, 127, y, stone);
    }
    // water pool bottom half
    for y in 80..127u16 {
        for x in 1..127u16 {
            put(&mut layer, x, y, water);
        }
    }
    // powder blobs above
    let mut r = Rng::new(99);
    for _ in 0..1500 {
        let x = 1 + (r.below(126) as u16);
        let y = 5 + (r.below(60) as u16);
        let m = match r.below(3) {
            0 => sand,
            1 => silt,
            _ => gold,
        };
        put(&mut layer, x, y, m);
    }
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }
    let before = powder_mass(&layer, &tab);
    let water_before = count(&layer, water);
    let mut layers = vec![layer];
    for tick in 1..=1200u64 {
        step_all_serial(&mut layers, &[], &tab, &l(), tick);
    }
    let after = powder_mass(&layers[0], &tab);
    println!(
        "D powder before={before} after={after} delta={}",
        after - before
    );
    println!(
        "D water before={water_before} after={}",
        count(&layers[0], water)
    );
    assert!(
        (before - after).abs() < 1e-6,
        "powder mass not conserved: {before} -> {after}"
    );
}

// ---------- E: a displaced liquid moves twice in one tick ----------
#[test]
fn e_displaced_liquid_moves_again_in_the_same_tick() {
    let tab = t();
    let water = id(&tab, "water");
    let grit = id(&tab, "grit"); // 1200 um: too coarse to entrain, displaces
    let stone = id(&tab, "stone");
    let mut layer = Layer::new(16, 16, LayerSlot::Plant, 7);
    for x in 0..16u16 {
        put(&mut layer, x, 15, stone);
    }
    // grit at (8,7), a single water cell directly under it at (8,8), floor of stone at (8,10)
    put(&mut layer, 8, 7, grit);
    put(&mut layer, 8, 8, water);
    for x in 0..16u16 {
        put(&mut layer, x, 9, stone);
    }
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }

    let mut layers = vec![layer];
    step_all_serial(&mut layers, &[], &tab, &l(), 2); // even tick -> ltr / primary +1
    let lay = &layers[0];
    // after one tick grit should be at (8,8) and the water at (8,7) - ONE cell each.
    let grit_pos: Vec<(u16, u16)> = (0..16)
        .flat_map(|y| (0..16).map(move |x| (x, y)))
        .filter(|&(x, y)| lay.mat[lay.idx(x, y)] == grit)
        .collect();
    let water_pos: Vec<(u16, u16)> = (0..16)
        .flat_map(|y| (0..16).map(move |x| (x, y)))
        .filter(|&(x, y)| lay.mat[lay.idx(x, y)] == water)
        .collect();
    println!("E grit={grit_pos:?} water={water_pos:?}");
    assert_eq!(
        water_pos,
        vec![(8, 7)],
        "the displaced water moved a second time in the same tick"
    );
}

// ---------- H: u16 overflow in chunk row arithmetic ----------
#[test]
fn h_tall_layer_does_not_overflow_chunk_arithmetic() {
    let tab = t();
    let mut layer = Layer::new(1, 65535, LayerSlot::Plant, 3);
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }
    let mut layers = vec![layer];
    step_all_serial(&mut layers, &[], &tab, &l(), 1);
}

// ---------- D2: same scene, coarse powder only (no entrainment path) ----------
#[test]
fn d2_scene_with_coarse_powder_only() {
    let tab = t();
    let (sand, grit, water, stone) = (
        id(&tab, "sand"),
        id(&tab, "grit"),
        id(&tab, "water"),
        id(&tab, "stone"),
    );
    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 0xABCD);
    for x in 0..128u16 {
        put(&mut layer, x, 127, stone);
        put(&mut layer, x, 0, stone);
    }
    for y in 0..128u16 {
        put(&mut layer, 0, y, stone);
        put(&mut layer, 127, y, stone);
    }
    for y in 80..127u16 {
        for x in 1..127u16 {
            put(&mut layer, x, y, water);
        }
    }
    let mut r = Rng::new(99);
    for _ in 0..1500 {
        let x = 1 + (r.below(126) as u16);
        let y = 5 + (r.below(60) as u16);
        let m = if r.bit() { sand } else { grit };
        put(&mut layer, x, y, m);
    }
    {
        let rr = layer.rect();
        layer.chunks.wake_rect(rr);
    }
    let before = powder_mass(&layer, &tab);
    let mut layers = vec![layer];
    for tick in 1..=1200u64 {
        step_all_serial(&mut layers, &[], &tab, &l(), tick);
    }
    let after = powder_mass(&layers[0], &tab);
    println!(
        "D2 powder before={before} after={after} delta={}",
        after - before
    );
    assert!(
        (before - after).abs() < 1e-6,
        "coarse-only powder mass: {before} -> {after}"
    );
}

// ---------- G: bed-load creep is scan-order biased (+x fast, -x one cell/tick) ------
#[test]
fn g_bed_load_creep_is_not_direction_symmetric() {
    let tab = t();
    let water = id(&tab, "water");
    let silt = id(&tab, "silt");
    let stone = id(&tab, "stone");
    // a flat stone bed with one row of water on it, dilute silt suspension in the middle
    let build = || {
        let mut layer = Layer::new(128, 32, LayerSlot::Plant, 11);
        for x in 0..128u16 {
            put(&mut layer, x, 20, stone);
        }
        for x in 1..127u16 {
            put(&mut layer, x, 19, water);
        }
        for x in 40..90u16 {
            let i = layer.idx(x, 19);
            layer.susp_mat[i] = silt;
            layer.susp_conc[i] = 4;
        }
        {
            let r = layer.rect();
            layer.chunks.wake_rect(r);
        }
        layer
    };
    let centroid = |l: &Layer| -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for x in 0..l.w {
            let i = l.idx(x, 19);
            let c = l.susp_conc[i] as f64 + if l.mat[i] == silt { 255.0 } else { 0.0 };
            num += c * x as f64;
            den += c;
        }
        if den == 0.0 { f64::NAN } else { num / den }
    };
    // even ticks only (dir = +1) vs odd ticks only (dir = -1): same number of steps
    let mut a = build();
    let c0 = centroid(&a);
    let mut layers = vec![a];
    for k in 0..40u64 {
        step_all_serial(&mut layers, &[], &tab, &l(), k * 2);
    } // all even
    let c_pos = centroid(&layers[0]);
    a = build();
    let mut layers2 = vec![a];
    for k in 0..40u64 {
        step_all_serial(&mut layers2, &[], &tab, &l(), k * 2 + 1);
    } // all odd
    let c_neg = centroid(&layers2[0]);
    println!(
        "G centroid start={c0:.2}  40 even ticks -> {c_pos:.2} (dx {:+.2})  40 odd ticks -> {c_neg:.2} (dx {:+.2})",
        c_pos - c0,
        c_neg - c0
    );
    let d_pos = (c_pos - c0).abs();
    let d_neg = (c_neg - c0).abs();
    assert!(
        (d_pos - d_neg).abs() < 0.25 * d_pos.max(d_neg).max(1.0),
        "bed-load transport is {d_pos:.2} cells one way and {d_neg:.2} the other"
    );
}

// ---------- C2: the other half of C - the drip precondition must be REACHABLE ------
// C only proves a drop is never minted for less than it costs. It would stay green if
// drip never fired at all, which is exactly the state the fix found it in: `step_powder`
// runs before the wet rules, so a powder cell with an empty cell beneath it has already
// fallen into it, and testing only the downward face meant no absorbed water ever
// returned to the world. This is the paired positive: a soaked pile at rest, in a
// sleeping chunk, with one flank open to air, sheds exactly one paid-for drop.
#[test]
fn c2_a_soaked_pile_at_rest_sheds_a_paid_for_drop() {
    let tab = t();
    let (sand, water, stone) = (id(&tab, "sand"), id(&tab, "water"), id(&tab, "stone"));
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 5);
    for x in 0..64u16 {
        put(&mut layer, x, 40, stone);
    }
    for x in 20..24u16 {
        put(&mut layer, x, 39, stone);
        put(&mut layer, x, 38, stone);
    }
    // soaked sand in a stone pocket: nothing can move, only the left flank sees air
    for y in 36..40u16 {
        for x in 24..30u16 {
            put(&mut layer, x, y, sand);
            let i = layer.idx(x, y);
            layer.wetness[i] = 255;
        }
    }
    for y in 36..40u16 {
        put(&mut layer, 30, y, stone);
    }
    {
        let r = layer.rect();
        layer.chunks.wake_rect(r);
    }
    let wet_before: u32 = layer.wetness.iter().map(|&w| w as u32).sum();

    let mut layers = vec![layer];
    for tick in 1..=1500u64 {
        step_all_serial(&mut layers, &[], &tab, &l(), tick);
    }
    let lay = &layers[0];
    let wet_after: u32 = lay.wetness.iter().map(|&w| w as u32).sum();
    let drops = count(lay, water);
    println!(
        "C2 wetness {wet_before} -> {wet_after} (-{}), drops {drops}, awake chunks {}",
        wet_before - wet_after,
        lay.chunks.awake_count()
    );
    assert!(
        drops > 0,
        "a soaked pile with a free face never shed a drop: absorption is a one-way sink"
    );
    // every drop is paid in full out of the patch, and the rest of the loss is evaporation
    assert!(
        wet_before - wet_after >= 896 * drops as u32,
        "{drops} drops cost only {} wetness units",
        wet_before - wet_after
    );
}
