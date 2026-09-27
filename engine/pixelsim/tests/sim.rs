//! Behavioural tests for the pixel simulation. Everything here runs headless and
//! serial: no Bevy app, no threads except where determinism-under-threading is the
//! thing being tested.

use pixelsim::*;

// ---- fixtures ---------------------------------------------------------------

fn table() -> MaterialTable {
    MaterialTable::embedded()
}

fn lut() -> ReposeLut {
    ReposeLut::analytic()
}

fn id(t: &MaterialTable, name: &str) -> u8 {
    t.id(name)
        .unwrap_or_else(|| panic!("material `{name}` missing from the table"))
        .0
}

fn tick_n(layers: &mut Vec<Layer>, t: &MaterialTable, l: &ReposeLut, n: u64) {
    for tick in 1..=n {
        step_all_serial(layers, &[], t, l, tick);
    }
}

/// A minimal but complete TOML entry, so tests can author tables the shipped data
/// does not contain (a second liquid, a bad table, ...).
#[allow(clippy::too_many_arguments)]
fn entry(
    name: &str,
    class: &str,
    density: f32,
    grain_density: f32,
    viscosity: f32,
    repose: f32,
    grain: f32,
    porosity: &str,
    hardness: f32,
) -> String {
    format!(
        r##"
[[material]]
name = "{name}"
class = "{class}"
color = "#808080"
density = {density}
grain_density = {grain_density}
viscosity = {viscosity}
repose_angle = {repose}
grain_size = {grain}
porosity = {porosity}
hardness = {hardness}
melt_pt = 0.0
boil_pt = 0.0
freeze_pt = 0.0
ignition_pt = 0.0
burn_products = []
thermal_cond = 0.0
heat_capacity = 0.0
elec_cond = 0.0
solubility = 0.0
reactivity_tags = []
"##
    )
}

fn air_entry() -> String {
    entry("air", "empty", 1.2, 0.0, 0.018, 0.0, 0.0, "inf", 0.0)
}

// ---- 1. material properties round-trip -------------------------------------

#[test]
fn material_table_round_trips_the_authored_data() {
    let t = table();

    // ids are positional and air is pinned at 0
    assert_eq!(t.id("air").unwrap().0, 0);
    assert_eq!(AIR.0, 0);
    assert!(t.is_empty(0));
    assert_eq!(t.class(0), MaterialClass::Empty);

    // every contract name resolves, and every entry survives the SoA mirror
    for name in [
        "air",
        "water",
        "sand",
        "silt",
        "grit",
        "gold_dust",
        "stone",
        "bedrock",
        "mesh",
        "filter_paper",
        "cardboard",
        "tin",
    ] {
        let mid = t.id(name).unwrap_or_else(|| panic!("missing {name}"));
        let m = t.get(mid);
        assert_eq!(m.name, name);
        assert_eq!(t.class(mid.0), m.class, "{name} class mirror");
        assert_eq!(t.density(mid.0), m.density, "{name} density mirror");
        assert_eq!(t.viscosity(mid.0), m.viscosity, "{name} viscosity mirror");
        assert_eq!(t.repose(mid.0), m.repose_angle, "{name} repose mirror");
        assert_eq!(t.grain(mid.0), m.grain_size, "{name} grain mirror");
        assert_eq!(t.porosity(mid.0), m.porosity, "{name} porosity mirror");
        assert_eq!(t.hardness(mid.0), m.hardness, "{name} hardness mirror");
    }
    assert_eq!(t.len(), 12);

    // hex colour deserialisation
    assert_eq!(t.get(t.id("sand").unwrap()).color, [0xc2, 0xb2, 0x80]);

    // derived liquid mobility: water disperses 5 cells and always moves
    let water = id(&t, "water");
    assert_eq!(t.dispersion(water), 5);
    assert_eq!(t.p_move(water), 1.0);

    // the sieve relation the vault calls the core loop
    let mesh = id(&t, "mesh");
    assert!(t.passes_powder(mesh, id(&t, "gold_dust")));
    // and the fine end of it: filter paper stops what mesh lets through, and its aperture
    // is authored independently of whether it soaks — that separation is the whole point of
    // `absorbency`, and it is what lets a filter pass water without being a sponge
    let paper = id(&t, "filter_paper");
    assert!(!t.passes_powder(paper, id(&t, "gold_dust")));
    assert!(!t.passes_powder(paper, id(&t, "silt")));
    assert!(t.passes_liquid(paper));
    assert_eq!(t.absorbency(paper), 0.0);
    assert!(t.passes_powder(mesh, id(&t, "silt")));
    assert!(!t.passes_powder(mesh, id(&t, "sand")));
    assert!(!t.passes_powder(mesh, id(&t, "grit")));
    assert!(t.passes_liquid(mesh));
    assert!(!t.passes_liquid(id(&t, "stone")));

    // undiggable rock
    assert!(dig_cost(&t, id(&t, "bedrock")).is_infinite());
    assert!(dig_cost(&t, id(&t, "stone")).is_finite());

    // wet-derived properties
    let sand = id(&t, "sand");
    assert_eq!(t.wet_density(sand, 0.0), 1600.0);
    assert_eq!(t.wet_density(sand, 1.0), 1890.0);
    assert_eq!(t.wet_repose(sand, 0.0), 34.0);
    assert_eq!(t.wet_repose(sand, 1.0), 48.0);

    // settle density prefers grain density where authored
    assert_eq!(t.density_grain_or_bulk(sand), 2650.0);
    assert_eq!(t.density_grain_or_bulk(id(&t, "stone")), 2600.0);
}

#[test]
fn material_table_rejects_bad_data() {
    // air must be entry 0
    let src = entry(
        "sand", "powder", 1600.0, 2650.0, 0.0, 34.0, 400.0, "120", 7.0,
    ) + &air_entry();
    assert!(matches!(
        MaterialTable::from_toml_str(&src),
        Err(MaterialError::AirNotFirst)
    ));

    // duplicate names
    let src = air_entry()
        + &entry(
            "sand", "powder", 1600.0, 2650.0, 0.0, 34.0, 400.0, "120", 7.0,
        )
        + &entry(
            "sand", "powder", 1600.0, 2650.0, 0.0, 34.0, 400.0, "120", 7.0,
        );
    assert!(matches!(
        MaterialTable::from_toml_str(&src),
        Err(MaterialError::DuplicateName(_))
    ));

    // powder repose angle out of range
    let src = air_entry()
        + &entry(
            "sand", "powder", 1600.0, 2650.0, 0.0, 89.0, 400.0, "120", 7.0,
        );
    assert!(matches!(
        MaterialTable::from_toml_str(&src),
        Err(MaterialError::BadValue(_))
    ));

    // zero density on a non-air entry
    let src = air_entry() + &entry("sand", "powder", 0.0, 2650.0, 0.0, 34.0, 400.0, "120", 7.0);
    assert!(matches!(
        MaterialTable::from_toml_str(&src),
        Err(MaterialError::BadValue(_))
    ));

    // a healthy two-entry table still loads
    let src = air_entry()
        + &entry(
            "sand", "powder", 1600.0, 2650.0, 0.0, 34.0, 400.0, "120", 7.0,
        );
    let t = MaterialTable::from_toml_str(&src).expect("valid table");
    assert_eq!(t.len(), 2);
}

// ---- 2. sand settles into a stable pile at the authored repose ---------------

#[test]
fn sand_settles_to_a_pile_at_its_authored_repose() {
    let t = table();
    let sand = id(&t, "sand");
    let target = t.repose(sand); // 34 degrees, authored

    let mut lut = ReposeLut::analytic();
    lut.calibrate(&t, &[target], 900, 0x5EED_0001);

    let layer = drop_pile(&t, &lut, sand, 900, 0x5EED_0001);
    let measured = ReposeLut::measure_pile_angle(&layer);
    println!("calibrated pile angle: {measured:.2} deg (target {target:.1})");
    assert!(
        (measured - target).abs() <= 4.0,
        "pile measured {measured:.2} deg, expected {target:.1} +- 4"
    );

    // and it is *stable*: no grain moves for a further 120 ticks
    let mut layers = vec![layer];
    let mut moved = 0u32;
    for tick in 10_000..10_120u64 {
        moved += step_all_serial(&mut layers, &[], &t, &lut, tick).moves;
    }
    assert_eq!(moved, 0, "settled pile is still moving");

    // a steeper powder must build a steeper pile than a shallower one
    let grit = id(&t, "grit"); // 38 deg
    let silt = id(&t, "silt"); // 30 deg
    let mut l2 = ReposeLut::analytic();
    l2.calibrate(&t, &[t.repose(grit), t.repose(silt)], 900, 7);
    let a_grit = ReposeLut::measure_pile_angle(&drop_pile(&t, &l2, grit, 900, 7));
    let a_silt = ReposeLut::measure_pile_angle(&drop_pile(&t, &l2, silt, 900, 7));
    println!("grit {a_grit:.2} deg vs silt {a_silt:.2} deg");
    assert!(
        a_grit > a_silt,
        "grit (38 deg) piled at {a_grit:.2}, silt (30 deg) at {a_silt:.2}"
    );
}

#[test]
fn wet_sand_holds_a_steeper_angle_than_dry_sand() {
    let t = table();
    let sand = id(&t, "sand");
    // wetness raises the effective repose angle, which lowers p_creep
    let lut = lut();
    let dry = lut.p_creep(t.wet_repose(sand, 0.0));
    let wet = lut.p_creep(t.wet_repose(sand, 1.0));
    assert!(wet < dry, "wet sand should creep less: {wet} vs {dry}");
    assert!(t.wet_density(sand, 1.0) > t.wet_density(sand, 0.0));
}

// ---- 3. water levels --------------------------------------------------------

/// surface height of every column, measured up from the floor
fn heights(layer: &Layer) -> Vec<u16> {
    (0..layer.w)
        .map(|x| {
            (0..layer.h)
                .find(|&y| layer.mat_at(x, y) != 0)
                .map(|y| layer.h - y)
                .unwrap_or(0)
        })
        .collect()
}

#[test]
fn water_levels_across_a_basin() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");

    let mut layer = Layer::new(128, 64, LayerSlot::Plant, 42);
    // a 32-wide, 32-deep block against the left wall
    for y in 32..64u16 {
        for x in 0..32u16 {
            layer.set_mat(x, y, water);
        }
    }
    let before: u32 = layer.mat.iter().filter(|&&m| m == water).count() as u32;

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 1500);
    let layer = &layers[0];

    let after: u32 = layer.mat.iter().filter(|&&m| m == water).count() as u32;
    assert_eq!(before, after, "water was created or destroyed");

    let h = heights(layer);
    let lo = *h.iter().min().unwrap();
    let hi = *h.iter().max().unwrap();
    let expected = before as u16 / layer.w; // 1024 / 128 = 8
    println!("levelled surface: min {lo}, max {hi}, expected {expected}");
    assert!(
        hi - lo <= 2,
        "surface is not level: spread {} (min {lo}, max {hi})",
        hi - lo
    );
    assert!(
        lo >= expected - 1 && hi <= expected + 1,
        "surface height {lo}..{hi} is not near the expected {expected}"
    );
    // it actually reached the far wall
    assert!(h[127] > 0, "water never reached the right-hand wall");
}

/// Communicating vessels: water poured into one arm of a U-tube must climb the *other*
/// arm until the two surfaces are level. The local rules can only move water down or
/// level, so before `equalize_levels` the filled arm just drained into the connecting
/// channel and sat there — the empty arm never rose, because nothing in the solver can
/// lift a cell up a walled shaft. This is the connected-component pass' whole reason to
/// exist, and the assertion is the level surface plus a hard mass-conservation check.
#[test]
fn water_climbs_the_far_arm_of_a_u_tube() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let stone = id(&t, "stone");

    let (w, h) = (64u16, 64u16);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, 7);

    // A U-tube: two vertical shafts joined by a channel under a central divider.
    let (left_wall, mid, right_wall) = (10u16, 32u16, 54u16);
    let (top, floor) = (12u16, 56u16); // floor row is `floor`; divider stops short of it
    for y in top..=floor {
        layer.set_mat(left_wall, y, stone);
        layer.set_mat(right_wall, y, stone);
    }
    for x in left_wall..=right_wall {
        layer.set_mat(x, floor, stone);
    }
    // central divider leaves the bottom four rows open as the connecting channel
    for y in top..floor - 4 {
        layer.set_mat(mid, y, stone);
    }

    // fill only the left shaft, well above the divider's foot
    for y in 20..floor {
        for x in left_wall + 1..mid {
            layer.set_mat(x, y, water);
        }
    }
    let before = layer.mat.iter().filter(|&&m| m == water).count() as u32;

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 3000);
    let layer = &layers[0];

    let after = layer.mat.iter().filter(|&&m| m == water).count() as u32;
    assert_eq!(
        before, after,
        "water was created or destroyed equalising a U-tube"
    );

    // surface row of each shaft = topmost water cell in it (smaller row = higher)
    let surface = |xs: std::ops::Range<u16>| -> Option<u16> {
        (0..h).find(|&y| xs.clone().any(|x| layer.mat_at(x, y) == water))
    };
    let left = surface(left_wall + 1..mid).expect("left shaft emptied entirely");
    let right = surface(mid + 1..right_wall).expect("water never climbed the right shaft");
    println!("U-tube surfaces: left row {left}, right row {right}");

    let spread = left.abs_diff(right);
    assert!(
        spread <= 2,
        "the arms never levelled: left surface row {left}, right {right} (spread {spread})"
    );
}

#[test]
fn water_falls_and_a_settled_pool_goes_to_sleep() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");

    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 3);
    for x in 0..128u16 {
        for y in 0..8u16 {
            layer.set_mat(x, y, water);
        }
    }
    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 400);

    // it fell to the floor
    assert_eq!(layers[0].mat_at(64, 127), water);
    assert_eq!(layers[0].mat_at(64, 0), 0);
    // and the whole layer is asleep, which is the entire performance argument
    assert_eq!(
        layers[0].chunks.awake_count(),
        0,
        "a settled pool must not keep its chunks awake"
    );
}

// ---- 4. density swap ordering ----------------------------------------------

#[test]
fn a_denser_liquid_sinks_below_a_lighter_one() {
    // the shipped table has only one liquid, so author a two-liquid table
    let src = air_entry()
        + &entry("light", "liquid", 800.0, 0.0, 1.0, 0.0, 0.0, "0", 0.0)
        + &entry("heavy", "liquid", 1600.0, 0.0, 1.0, 0.0, 0.0, "0", 0.0)
        + &entry("wall", "solid", 2600.0, 0.0, 0.0, 0.0, 0.0, "0", 6.0);
    let t = MaterialTable::from_toml_str(&src).expect("two-liquid table");
    let lut = lut();
    let light = id(&t, "light");
    let heavy = id(&t, "heavy");
    let wall = id(&t, "wall");

    // a sealed 1-wide column: heavy on top, light underneath. Without the walls the
    // liquids would simply spread out over the floor, which is a different test.
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 11);
    for y in 30..64u16 {
        layer.set_mat(29, y, wall);
        layer.set_mat(31, y, wall);
    }
    for y in 32..48u16 {
        layer.set_mat(30, y, heavy);
    }
    for y in 48..64u16 {
        layer.set_mat(30, y, light);
    }

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 300);
    let layer = &layers[0];

    // after settling, every heavy cell is below every light cell in the column
    let lowest_light = (32..64u16)
        .filter(|&y| layer.mat_at(30, y) == light)
        .max()
        .expect("light survived");
    let highest_heavy = (32..64u16)
        .filter(|&y| layer.mat_at(30, y) == heavy)
        .min()
        .expect("heavy survived");
    println!("lowest light row {lowest_light}, highest heavy row {highest_heavy}");
    assert!(
        highest_heavy > lowest_light,
        "heavy ({highest_heavy}) did not sink below light ({lowest_light})"
    );
    // nothing was created or lost
    assert_eq!(layer.mat.iter().filter(|&&m| m == heavy).count(), 16);
    assert_eq!(layer.mat.iter().filter(|&&m| m == light).count(), 16);
}

#[test]
fn a_powder_displaces_water_it_outweighs() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let grit = id(&t, "grit"); // 1200 µm, too coarse to entrain

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 5);
    for y in 40..64u16 {
        for x in 20..44u16 {
            layer.set_mat(x, y, water);
        }
    }
    layer.set_mat(32, 30, grit);

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 200);
    let layer = &layers[0];

    // the grain sank through the water rather than floating on it
    let grit_y = (0..64u16)
        .find(|&y| (0..64u16).any(|x| layer.mat_at(x, y) == grit))
        .expect("grit survived");
    println!("grit came to rest at row {grit_y}");
    assert!(
        grit_y >= 60,
        "grit should have sunk to the floor, got {grit_y}"
    );
}

// ---- 5. determinism ---------------------------------------------------------

fn chaos_layer(t: &MaterialTable, seed: u64) -> Layer {
    let mut layer = Layer::new(192, 128, LayerSlot::Plant, seed);
    let sand = id(t, "sand");
    let water = id(t, "water");
    let stone = id(t, "stone");
    // a deterministic scatter — hash_rng, so the scene itself adds no randomness
    for y in 0..128u16 {
        for x in 0..192u16 {
            let h = hash_rng(0xABCD, 0, x, y);
            let m = match h % 16 {
                0..=3 => sand,
                4..=5 => water,
                6 => stone,
                _ => 0,
            };
            if m != 0 {
                layer.set_mat(x, y, m);
            }
        }
    }
    layer
}

#[test]
fn same_seed_and_input_give_an_identical_grid() {
    let t = table();
    let lut = lut();

    let mut a = vec![chaos_layer(&t, 0xDEAD_BEEF)];
    let mut b = vec![chaos_layer(&t, 0xDEAD_BEEF)];
    assert_eq!(a[0].hash(), b[0].hash(), "initial grids differ");

    tick_n(&mut a, &t, &lut, 120);
    tick_n(&mut b, &t, &lut, 120);
    assert_eq!(a[0].hash(), b[0].hash(), "same seed diverged");

    // a different seed must actually change the outcome, or the test above is vacuous
    let mut c = vec![chaos_layer(&t, 0x1234_5678)];
    tick_n(&mut c, &t, &lut, 120);
    assert_ne!(a[0].hash(), c[0].hash(), "seed has no effect");
}

#[test]
fn threading_does_not_perturb_the_result() {
    let t = table();
    let lut = lut();
    let pool = bevy::tasks::TaskPool::new();

    let mut serial: Vec<Layer> = (0..3)
        .map(|i| chaos_layer(&t, 0xF00D_0000 + i as u64))
        .collect();
    let mut parallel: Vec<Layer> = (0..3)
        .map(|i| chaos_layer(&t, 0xF00D_0000 + i as u64))
        .collect();

    for tick in 1..=80u64 {
        step_all_serial(&mut serial, &[], &t, &lut, tick);
        step_all(&mut parallel, &[], &t, &lut, tick, &pool);
    }
    for i in 0..3 {
        assert_eq!(
            serial[i].hash(),
            parallel[i].hash(),
            "layer {i} diverged between the serial and pooled step"
        );
    }
}

#[test]
fn the_rng_is_the_documented_stream() {
    // xorshift64*: fixed, so a change to the generator is a visible test failure
    let mut r = Rng::new(1);
    let first: Vec<u64> = (0..3).map(|_| r.next_u64()).collect();
    let mut r2 = Rng::new(1);
    let again: Vec<u64> = (0..3).map(|_| r2.next_u64()).collect();
    assert_eq!(first, again);

    // hash_rng is stateless and order-independent
    assert_eq!(hash_rng(7, 9, 3, 4), hash_rng(7, 9, 3, 4));
    assert_ne!(hash_rng(7, 9, 3, 4), hash_rng(7, 9, 4, 3));

    // below() stays in range, chance_u8(0) never fires
    let mut r = Rng::new(0xABCD);
    for _ in 0..1000 {
        assert!(r.below(7) < 7);
        assert!(!r.chance_u8(0));
        let f = r.f32();
        assert!((0.0..1.0).contains(&f));
    }
}

// ---- 6. dirty-chunk / activation correctness -------------------------------

#[test]
fn a_move_across_a_chunk_boundary_wakes_the_next_chunk() {
    let t = table();
    let lut = lut();
    let sand = id(&t, "sand");

    // 128x128 = 2x2 chunks of 64px
    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 1);
    // one grain already resting on the floor, in the lower-left chunk
    layer.set_mat(10, 127, sand);
    let mut layers = vec![layer];

    // let everything go quiet
    tick_n(&mut layers, &t, &lut, 60);
    assert_eq!(
        layers[0].chunks.awake_count(),
        0,
        "layer never went to sleep"
    );

    let upper = layers[0].chunks.chunk_of(10, 60);
    let lower = layers[0].chunks.chunk_of(10, 70);
    assert_ne!(upper, lower, "test needs two distinct chunk rows");

    // dropping a grain into the upper chunk must wake that chunk
    layers[0].set_mat(10, 60, sand);
    assert!(
        layers[0].chunks.next.get(upper),
        "set_mat did not wake the containing chunk"
    );

    // it has to fall the whole way: crossing y=63 -> y=64 requires the lower chunk to
    // be woken by the mover, not by anything else
    tick_n(&mut layers, &t, &lut, 200);
    assert_eq!(
        layers[0].mat_at(10, 60),
        0,
        "grain never left its start cell"
    );
    // both grains ended up on the floor row: the dropped one crossed the y=63 seam,
    // then the diagonal rule slid it off the resting grain
    let topmost = (0..128u16)
        .find(|&y| (0..128u16).any(|x| layers[0].mat_at(x, y) == sand))
        .expect("sand survived");
    assert_eq!(
        topmost, 127,
        "grain stalled at row {topmost} instead of reaching the floor across the chunk seam"
    );
    assert_eq!(layers[0].mat.iter().filter(|&&m| m == sand).count(), 2);
    assert_eq!(layers[0].chunks.awake_count(), 0);
}

#[test]
fn a_boundary_cell_wakes_its_eight_neighbours() {
    let mut cm = ChunkMap::new(192, 192); // 3x3 chunks
    cm.next.clear_all();
    // (64, 64) is the first cell of the centre chunk — a corner cell, so all eight
    // neighbours must come along
    cm.wake(64, 64);
    for cy in 0..3usize {
        for cx in 0..3usize {
            assert!(
                cm.next.get(cy * 3 + cx),
                "chunk ({cx},{cy}) was not woken by a corner cell"
            );
        }
    }

    // an interior cell wakes only its own chunk
    let mut cm = ChunkMap::new(192, 192);
    cm.next.clear_all();
    cm.wake(96, 96);
    assert!(cm.next.get(4));
    assert_eq!(cm.next.count(), 1);
}

#[test]
fn dirty_chunks_are_reported_once_and_then_cleared() {
    let t = table();
    let lut = lut();
    let sand = id(&t, "sand");
    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 1);

    // construction dirties everything so the first bake is a full bake
    assert_eq!(layer.chunks.take_dirty().len(), 4);
    assert!(layer.chunks.take_dirty().is_empty());

    layer.set_mat(100, 100, sand);
    assert_eq!(layer.chunks.take_dirty(), vec![3]);
    assert!(layer.chunks.take_dirty().is_empty());

    // a solver move dirties both the source and the destination chunk
    layer.set_mat(70, 63, sand);
    layer.chunks.take_dirty();
    let mut layers = vec![layer];
    step_all_serial(&mut layers, &[], &t, &lut, 1);
    let dirty = layers[0].chunks.take_dirty();
    assert!(dirty.contains(&1), "source chunk not dirtied: {dirty:?}");
    assert!(
        dirty.contains(&3),
        "destination chunk not dirtied: {dirty:?}"
    );
}

#[test]
fn sculpting_wakes_and_dirties_the_touched_rect() {
    let t = table();
    let mut layer = Layer::new(256, 256, LayerSlot::Plant, 1);
    let mut journal = UndoJournal::new();
    // put it all to sleep first
    layer.chunks.next.clear_all();
    layer.chunks.begin_tick();
    layer.chunks.dirty.clear_all();
    assert_eq!(layer.chunks.awake_count(), 0);

    let sand = t.id("sand").unwrap();
    let rect = apply_brush(
        &mut layer,
        &mut journal,
        0,
        &t,
        (200, 200),
        Brush::Round { r: 6 },
        Tool::Paint(sand),
        Mirror::default(),
    );
    assert!(rect.contains(200, 200));
    assert_eq!(layer.mat_at(200, 200), sand.0);
    assert!(
        layer.chunks.awake_count() > 0,
        "sculpt did not wake a chunk"
    );
    assert!(
        layer
            .chunks
            .take_dirty()
            .contains(&layer.chunks.chunk_of(200, 200)),
        "sculpt did not dirty its own chunk"
    );
}

// ---- sieving, ports, wetness, slurry, undo ----------------------------------

#[test]
fn a_mesh_sieves_by_grain_size() {
    let t = table();
    let lut = lut();
    let mesh = id(&t, "mesh");
    let sand = id(&t, "sand");
    let gold = id(&t, "gold_dust");

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 2);
    // a mesh shelf across the middle
    for x in 0..64u16 {
        layer.set_mat(x, 32, mesh);
    }
    // coarse sand on the left half of the shelf, fine gold dust on the right
    for x in 8..24u16 {
        layer.set_mat(x, 31, sand);
    }
    for x in 40..56u16 {
        layer.set_mat(x, 31, gold);
    }

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 200);
    let layer = &layers[0];

    let sand_below = (33..64u16)
        .flat_map(|y| (0..64u16).map(move |x| (x, y)))
        .filter(|&(x, y)| layer.mat_at(x, y) == sand)
        .count();
    let gold_below = (33..64u16)
        .flat_map(|y| (0..64u16).map(move |x| (x, y)))
        .filter(|&(x, y)| layer.mat_at(x, y) == gold)
        .count();
    println!("through the mesh: sand {sand_below}, gold_dust {gold_below}");
    assert_eq!(sand_below, 0, "sand (400 µm) passed a 250 µm mesh");
    assert!(gold_below > 8, "gold_dust (60 µm) did not pass the mesh");
    // the mesh itself is intact
    assert_eq!(
        (0..64u16).filter(|&x| layer.mat_at(x, 32) == mesh).count(),
        64
    );
}

#[test]
fn a_port_moves_one_cell_per_tick_and_conserves_it() {
    let t = table();
    let sand = id(&t, "sand");
    let mut layers = vec![
        Layer::new(64, 64, LayerSlot::Plant, 1),
        Layer::new(64, 64, LayerSlot::Plant, 2),
    ];
    layers[0].set_mat(5, 5, sand);
    let src_i = layers[0].idx(5, 5);
    layers[0].wetness[src_i] = 200;

    let ports = [PortPair {
        from_layer: 0,
        from: (5, 5),
        to_layer: 1,
        to: (9, 9),
        per_tick: 1,
    }];

    assert_eq!(apply_ports(&mut layers, &ports, &t), 1);
    assert_eq!(layers[0].mat_at(5, 5), 0);
    assert_eq!(layers[1].mat_at(9, 9), sand);
    // aux channels travel with the cell, and the port bit is set
    assert_eq!(layers[1].wetness[layers[1].idx(9, 9)], 200);
    assert!(layers[1].flags[layers[1].idx(9, 9)] & FLAG_PORT != 0);

    // an empty source transfers nothing; a blocked destination transfers nothing
    assert_eq!(apply_ports(&mut layers, &ports, &t), 0);
    layers[0].set_mat(5, 5, sand);
    assert_eq!(apply_ports(&mut layers, &ports, &t), 0);
}

#[test]
fn water_wets_sand_and_wetness_spreads() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let sand = id(&t, "sand");

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 4);
    for y in 40..64u16 {
        for x in 0..64u16 {
            layer.set_mat(x, y, sand);
        }
    }
    for y in 30..36u16 {
        for x in 28..36u16 {
            layer.set_mat(x, y, water);
        }
    }
    let water_before = layer.mat.iter().filter(|&&m| m == water).count();

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 200);
    let layer = &layers[0];

    let water_after = layer.mat.iter().filter(|&&m| m == water).count();
    let wet_cells = layer.wetness.iter().filter(|&&w| w > 0).count();
    let total_wet: u32 = layer.wetness.iter().map(|&w| w as u32).sum();
    println!(
        "water {water_before} -> {water_after}, wet cells {wet_cells}, wetness units {total_wet}"
    );
    assert!(water_after < water_before, "no water was absorbed");
    assert!(
        wet_cells > 8,
        "wetness did not spread past the contact face"
    );
    assert!(total_wet > 0);
    // the wetting front is under the puddle, not above it
    assert!(layer.wetness[layer.idx(32, 40)] > 0);
}

/// The thing anybody expects on first contact with wet sand: pour water on a deep dry bed
/// and it soaks in until it is gone.
///
/// It used not to. A pool wetted the handful of cells it touched and then sat there for
/// ever, because the only downward transport was capillary diffusion — which equilibrates
/// as soon as every step of the gradient falls under its dead-band, leaving a shallow damp
/// patch — and with the contact patch full, absorption's capacity guard refused every
/// further cell. Measured: a 160-cell pool over a bed with room for 984 stalled at 84 cells
/// left, having used an eighth of the bed's capacity, and stayed there for 4000 ticks.
/// `wet::drain_cell` is the missing rule.
#[test]
fn a_pool_seeps_into_a_deep_bed_until_it_is_gone() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let sand = id(&t, "sand");
    let bedrock = id(&t, "bedrock");

    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 7);
    for y in 80..116u16 {
        for x in 16..112u16 {
            layer.set_mat(x, y, sand);
        }
    }
    for y in 74..79u16 {
        for x in 48..80u16 {
            layer.set_mat(x, y, water);
        }
    }
    for y in 116..128u16 {
        for x in 0..128u16 {
            layer.set_mat(x, y, bedrock);
        }
    }
    let poured = layer.mat.iter().filter(|&&m| m == water).count() as u32;
    // the bed can hold far more than the pool, so nothing here is capacity-limited
    let sand_cells = layer.mat.iter().filter(|&&m| m == sand).count() as u32;
    let capacity = sand_cells * 255 / ABSORB_UNITS;
    assert!(capacity > poured * 4, "bed too small to be a fair test");

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 300);
    let layer = &layers[0];

    let left = layer.mat.iter().filter(|&&m| m == water).count() as u32;
    let held: u32 = layer.wetness.iter().map(|&w| w as u32).sum();
    println!(
        "poured {poured} cells, {left} left standing, {} cells' worth held as wetness",
        held / ABSORB_UNITS
    );
    assert_eq!(left, 0, "the pool never finished soaking in");
    // and the water is still there, as wetness: absorption must not be a way to delete it
    let expected = poured * ABSORB_UNITS;
    assert!(
        held as f32 > expected as f32 * 0.95,
        "absorbed {expected} units of water, {held} survived"
    );
    // it went DOWN, which is the rule that was missing: the front reaches the bedrock floor
    let front = (0..128u16)
        .rev()
        .find(|&y| (0..128u16).any(|x| layer.wetness[layer.idx(x, y)] > 0))
        .unwrap();
    assert!(
        front >= 114,
        "wetting front only reached row {front} of 115"
    );
}

/// Seepage has to work on a scene with no motion in it at all, because that is the scene it
/// happens in: water sitting still on sand that has finished settling.
///
/// Chunk sleep is a *motion* predicate, so a still pool of water resting on sand puts every
/// chunk to sleep within `SLEEP_TICKS` and never wakes them. The wet rules ride the damp set
/// for exactly this reason; absorption and capillary used to ride the awake set, which meant
/// a still pool was a permanently still pool.
///
/// The water sits *on* the sand under a stone lid rather than buried inside it: sand fine
/// enough to entrain (see `ENTRAIN_GRAIN_UM`) clouds into any water directly below or
/// diagonally below it, so a bare pocket surrounded by sand is no longer motionless — the
/// grains above it would rain in. The lid puts a non-powder ceiling over the pool so nothing
/// can fall into it, which is the still scene this test is actually about.
#[test]
fn a_sealed_pool_still_soaks_into_the_sand_around_it() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let sand = id(&t, "sand");
    let stone = id(&t, "stone");

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 11);
    for y in 8..56u16 {
        for x in 8..56u16 {
            layer.set_mat(x, y, sand);
        }
    }
    // a stone lid one cell wider than the pool on each side, so no grain sits above or
    // diagonally above the water and nothing can fall into it
    for x in 19..=45u16 {
        layer.set_mat(x, 29, stone);
    }
    // the pool: a strip resting on the sand under the lid, no free face, so nothing moves
    for x in 20..45u16 {
        layer.set_mat(x, 30, water);
    }
    let poured = layer.mat.iter().filter(|&&m| m == water).count();

    let mut layers = vec![layer];
    // long enough that the motion fuse has burnt out many times over
    tick_n(&mut layers, &t, &lut, 400);
    let layer = &layers[0];

    println!(
        "sealed pool {poured} -> {} cells, awake chunks {}, damp chunks {}",
        layer.mat.iter().filter(|&&m| m == water).count(),
        layer.chunks.awake_count(),
        layer.chunks.damp_count(),
    );
    assert_eq!(
        layer.mat.iter().filter(|&&m| m == water).count(),
        0,
        "a sealed pool never soaked in — the wet pass is riding the awake set again"
    );
    assert!(layer.wetness.iter().any(|&w| w > 0));
}

/// Drainage is gravity, not diffusion: it moves at a stated rate, it stops at field
/// capacity, and it leaves the medium damp rather than draining it dry. Driven through
/// `wet::drain` alone, so capillary spread cannot be mistaken for it.
#[test]
fn drainage_is_gravity_and_stops_at_field_capacity() {
    let t = table();
    let sand = id(&t, "sand");

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 12);
    for y in 20..60u16 {
        for x in 20..44u16 {
            layer.set_mat(x, y, sand);
        }
    }
    // one saturated cell over dry sand, and one already sitting at field capacity
    let top = layer.idx(30, 20);
    layer.wetness[top] = 255;
    let held = layer.idx(36, 20);
    layer.wetness[held] = FIELD_CAPACITY;
    let r = CellRect {
        x0: 0,
        y0: 0,
        x1: 63,
        y1: 63,
    };
    layer.chunks.wake_rect(r);

    // one pass: exactly one transfer of the documented size, one row down and no further
    drain(&mut layer, &t);
    assert_eq!(layer.wetness[layer.idx(30, 21)], DRAIN_MAX_TRANSFER);
    assert_eq!(layer.wetness[layer.idx(30, 20)], 255 - DRAIN_MAX_TRANSFER);
    assert_eq!(
        layer.wetness[layer.idx(30, 22)],
        0,
        "water fell more than one row in a tick"
    );
    // and gravity has no claim on water held at field capacity
    assert_eq!(
        layer.wetness[layer.idx(36, 21)],
        0,
        "water at field capacity drained anyway"
    );

    for _ in 0..200 {
        layer.chunks.wake_rect(r);
        drain(&mut layer, &t);
    }
    let column: Vec<u8> = (20..30u16)
        .map(|y| layer.wetness[layer.idx(30, y)])
        .collect();
    println!("column after draining: {column:?}");
    // no cell holds more than gravity allows...
    for (k, &w) in column.iter().enumerate() {
        assert!(
            w <= FIELD_CAPACITY,
            "row {} kept {w} units, above field capacity",
            20 + k
        );
    }
    // ...every unit is still in the column — drainage moves water, it does not spend it...
    let total: u32 = column.iter().map(|&w| w as u32).sum();
    assert_eq!(
        total, 255,
        "column holds {total} of the 255 units it started with"
    );
    // ...and it left a damp trail rather than dumping everything on one row
    let damp_rows = column.iter().filter(|&&w| w > 0).count();
    assert!(
        damp_rows >= 5,
        "only {damp_rows} rows ended up damp: {column:?}"
    );
    assert_eq!(
        layer.wetness[layer.idx(36, 20)],
        FIELD_CAPACITY,
        "the field-capacity cell lost water to gravity"
    );
}

/// The wet passes must cost nothing in a vessel with no water in it, or every dry scene pays
/// for the wet rules for ever. The damp set is pruned as it is scanned, so this converges.
#[test]
fn a_dry_vessel_prunes_itself_out_of_the_wet_pass() {
    let t = table();
    let lut = lut();
    let stone = id(&t, "stone");

    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 13);
    layer.fill(stone);
    let r = CellRect {
        x0: 0,
        y0: 0,
        x1: 127,
        y1: 127,
    };
    layer.chunks.wake_rect(r);

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 64);
    assert_eq!(
        layers[0].chunks.damp_count(),
        0,
        "a dry vessel is still paying for the wet pass"
    );
}

/// A settled flooded vessel is the worst case for rules that cannot sleep on motion: every
/// chunk holds water against sand for ever, so there is always something to look at and
/// never anything to do. Unfused it measured 2.95 ms a tick — 18% of the frame budget, on a
/// scene where nothing can change. The wet fuse drops an idle chunk to one visit in
/// `WET_IDLE_EVERY` ticks, and any edit re-arms it.
#[test]
#[cfg_attr(debug_assertions, ignore = "wall-clock; run in --release")]
fn a_settled_flood_is_cheap_to_keep_wet() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let sand = id(&t, "sand");

    let mut layer = Layer::new(1024, 512, LayerSlot::Plant, 3);
    for y in 256..512u16 {
        for x in 0..1024u16 {
            layer.set_mat(x, y, sand);
            let i = layer.idx(x, y);
            layer.wetness[i] = 255;
        }
    }
    for y in 96..256u16 {
        for x in 0..1024u16 {
            layer.set_mat(x, y, water);
        }
    }
    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 400); // let sleep and the fuse converge

    let mut wet_ms = Vec::new();
    for tick in 401..=600u64 {
        let s = step_all_serial(&mut layers, &[], &t, &lut, tick);
        wet_ms.push(s.wet_ms);
    }
    wet_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = wet_ms[wet_ms.len() / 2];
    println!("settled flood: wet pass {median:.3} ms/tick median");
    assert!(
        median < 1.0,
        "the wet pass costs {median:.2} ms/tick on a scene that cannot change"
    );
}

#[test]
fn fine_powder_entrains_into_water_and_settles_back_out() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let gold = id(&t, "gold_dust");

    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 6);
    for y in 32..64u16 {
        for x in 20..44u16 {
            layer.set_mat(x, y, water);
        }
    }
    for x in 28..36u16 {
        layer.set_mat(x, 30, gold);
    }

    let mut layers = vec![layer];
    // Early on, the grains meet the water surface and go into suspension. The bound is
    // deliberately loose: it used to be 12 ticks, which was the exact tick the grains
    // happened to arrive on, so adding the liquid solver's diagonal step — which changes
    // how the surface reorganises under them, and nothing about entrainment — moved first
    // contact to tick 14 and failed a test whose subject is *whether* fine powder
    // entrains, not when. Measured: first suspension at 14, fully laden by 16.
    tick_n(&mut layers, &t, &lut, 30);
    let suspended: u32 = layers[0].susp_conc.iter().map(|&c| c as u32).sum();
    println!("suspended concentration after 30 ticks: {suspended}");
    assert!(suspended > 0, "fine powder never entrained");

    // and then settles out again as a wet deposit at the bottom
    tick_n(&mut layers, &t, &lut, 600);
    let layer = &layers[0];
    let deposited = layer.mat.iter().filter(|&&m| m == gold).count();
    println!("gold cells re-deposited: {deposited}");
    assert!(deposited > 0, "suspension never deposited");
    let lowest = (0..64u16)
        .rev()
        .find(|&y| (0..64u16).any(|x| layer.mat_at(x, y) == gold))
        .unwrap();
    assert!(lowest > 50, "deposit did not sink, lowest row {lowest}");
}

/// A cardboard cup of water soaks, weakens and eventually gives way.
///
/// Before `absorbency` existed, `absorbent()` was `Powder && porosity > 0`, so no solid
/// could hold a single unit of water and cardboard was perfectly waterproof — a cup of
/// water sat in it forever. `porosity` could not be reused as the flag either: it is a
/// sieve aperture, so the test would have made steel mesh (250 µm) a sponge while leaving
/// cardboard (no aperture at all) bone dry. Hence a separate property, asserted below on
/// both materials.
#[test]
fn a_cardboard_cup_goes_soggy_and_gives_way() {
    let t = table();
    let lut = lut();
    let (water, card, rock, mesh) = (
        id(&t, "water"),
        id(&t, "cardboard"),
        id(&t, "bedrock"),
        id(&t, "mesh"),
    );
    // mesh sieves but must never soak — the distinction `absorbency` exists to draw
    assert!(t.get(t.id("mesh").unwrap()).porosity > 0.0);
    assert_eq!(t.get(t.id("mesh").unwrap()).absorbency, 0.0);
    assert_eq!(t.get(t.id("cardboard").unwrap()).porosity, 0.0);
    assert!(t.get(t.id("cardboard").unwrap()).absorbency > 0.0);

    let (w, h) = (48u16, 64u16);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, 11);
    for x in 0..w {
        layer.set_mat(x, h - 1, rock);
    }
    // a cardboard cup, floor at y=30, hanging over open air
    let (x0, x1, floor) = (16u16, 32u16, 30u16);
    for x in x0..=x1 {
        for dy in 0..2u16 {
            layer.set_mat(x, floor + dy, card);
        }
    }
    for y in 18..floor {
        for dx in 0..2u16 {
            layer.set_mat(x0 + dx, y, card);
            layer.set_mat(x1 - dx, y, card);
        }
    }
    // a lump of mesh sitting in the water, as a control
    for x in 20..24u16 {
        layer.set_mat(x, floor - 1, mesh);
    }
    // fill the cup
    for y in 20..floor {
        for x in (x0 + 2)..=(x1 - 2) {
            if layer.mat_at(x, y) == 0 {
                layer.set_mat(x, y, water);
            }
        }
    }

    let below_the_cup = |l: &Layer| {
        (floor + 2..h - 1)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| l.mat_at(x, y) == water)
            .count()
    };
    let count = |l: &Layer, m: u8| l.mat.iter().filter(|&&v| v == m).count();
    let wettest = |l: &Layer, m: u8| {
        (0..l.mat.len())
            .filter(|&i| l.mat[i] == m)
            .map(|i| l.wetness[i])
            .max()
            .unwrap_or(0)
    };
    let worst_wear = |l: &Layer| {
        (0..l.mat.len())
            .filter(|&i| l.mat[i] == card)
            .map(|i| l.wear[i])
            .max()
            .unwrap_or(0)
    };

    let water_units = |l: &Layer| -> u64 {
        let cells = l.mat.iter().filter(|&&m| m == water).count() as u64;
        let held: u64 = l.wetness.iter().map(|&v| v as u64).sum();
        cells * ABSORB_UNITS as u64 + held
    };
    let card0 = count(&layer, card);
    let w0 = water_units(&layer);
    let mut layers = vec![layer];

    tick_n(&mut layers, &t, &lut, 60);
    let soaked = wettest(&layers[0], card);
    println!("cardboard wetness after 60 ticks: {soaked}");
    assert!(soaked > 0, "cardboard never got wet at all");
    assert_eq!(
        wettest(&layers[0], mesh),
        0,
        "mesh soaked up water — absorbency is being confused with sieve aperture"
    );

    tick_n(&mut layers, &t, &lut, 240);
    let wear = worst_wear(&layers[0]);
    println!("worst cardboard wear after 300 ticks: {wear}");
    assert!(wear > 0, "soaked cardboard never weakened");

    // and eventually the floor gives way and the water goes through it
    let mut burst_at = None;
    // the deepest cardboard cell anywhere, i.e. how far the floor has drooped
    let lowest = |l: &Layer| {
        (0..h)
            .rev()
            .find(|&y| (0..w).any(|x| l.mat_at(x, y) == card))
    };
    let floor0 = lowest(&layers[0]).unwrap();

    for tick in 301..=4000u64 {
        step_all_serial(&mut layers, &[], &t, &lut, tick);
        if burst_at.is_none() && lowest(&layers[0]) > Some(floor0) {
            burst_at = Some(tick);
        }
    }
    let leaked = below_the_cup(&layers[0]);
    let droop = lowest(&layers[0]).unwrap() - floor0;
    println!(
        "cardboard {card0} -> {} cells, first sag at tick {burst_at:?}, drooped {droop} rows, {leaked} water cells below the cup",
        count(&layers[0], card)
    );

    // Deformation, not destruction. The first cut deleted cells at full wear, which loses
    // material and reads as the box being eaten rather than going soft.
    assert_eq!(
        count(&layers[0], card),
        card0,
        "cardboard cells went missing — a soggy box must displace material, not lose it"
    );
    assert!(burst_at.is_some(), "the soaked board never sagged");
    assert!(droop > 0, "the floor did not droop");
    assert!(
        leaked > 0,
        "the floor sagged away but no water came through"
    );

    // The layer that touches the water must not be the stiff one.
    //
    // It was: the cup hollowed out from the far side while the wetted face stayed perfectly
    // intact, because drainage pulled every cell to the granular FIELD_CAPACITY of 48,
    // below SOG_MIN_WETNESS, so wear froze everywhere except the bottom-most cell of each
    // column, which had nowhere left to drain to. Field capacity now follows `absorbency`,
    // so soaked board stays soaked instead of acting as a drain pipe. Measured on wear,
    // which is the thing that was freezing.
    let row_wear = |y: u16| {
        (x0..=x1)
            .filter(|&x| layers[0].mat_at(x, y) == card)
            .map(|x| layers[0].wear[layers[0].idx(x, y)] as u32)
            .max()
            .unwrap_or(0)
    };
    println!(
        "worst wear: contact row {}, outer row {}",
        row_wear(floor),
        row_wear(floor + 1)
    );
    // A margin, not an ordering: both rows saturate, and which reaches 255 a couple of aux
    // windows sooner is scheduling noise — 252 against 255 on one run. What matters is that
    // the contact row goes thoroughly soft rather than freezing part-way, which is exactly
    // what the old field capacity did to it: pinned at the wetness 48 units buys, far below
    // SOG_MIN_WETNESS, where its wear stopped and it stayed sound and immovable for ever.
    assert!(
        row_wear(floor) >= SAG_MIN_WEAR as u32,
        "the water-facing layer never went soft: wear {} against the {} needed to sag",
        row_wear(floor),
        SAG_MIN_WEAR
    );

    // Sagging is a swap, so it moves no water at all; what is lost here is evaporation over
    // 4000 ticks from a cup with a free surface.
    let lost = w0.saturating_sub(water_units(&layers[0]));
    println!(
        "water lost over 4000 ticks: {:.1} cells of {:.0}",
        lost as f64 / ABSORB_UNITS as f64,
        w0 as f64 / ABSORB_UNITS as f64
    );
    assert!(
        lost < 40 * ABSORB_UNITS as u64,
        "{:.0} cells of water went missing — sagging should move no water",
        lost as f64 / ABSORB_UNITS as f64
    );
}

/// Slurry poured onto filter paper leaves the solids behind and lets the water through.
///
/// The sieve relation was already here and already right — `passes_powder` is
/// `grain <= porosity` — but it only ever filtered what fell on the screen *dry*. Suspended
/// solids travel as passengers of a liquid cell, so percolation asked the cell whether it
/// fitted through the aperture and never asked the grain: with an 8 µm paper that should
/// retain every 60 µm grain, gold washed straight through in the water.
///
/// What builds the cake is rules that were already there. A laden cell that cannot descend
/// is at its bed, so `settle_at_bed` concentrates it until it holds a whole grain and
/// deposits it above the paper.
#[test]
fn filter_paper_keeps_the_solids_and_passes_the_water() {
    let t = table();
    let lut = lut();
    let (water, gold, paper, rock) = (
        id(&t, "water"),
        id(&t, "gold_dust"),
        id(&t, "filter_paper"),
        id(&t, "bedrock"),
    );
    let (w, h) = (40u16, 40u16);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, 5);
    for x in 0..w {
        layer.set_mat(x, h - 1, rock);
    }
    // a funnel: rock walls down to a paper across the throat, open space beneath
    let sheet = 16u16;
    for y in 0..sheet {
        for x in [0u16, 1, w - 2, w - 1] {
            layer.set_mat(x, y, rock);
        }
    }
    for x in 2..w - 2 {
        layer.set_mat(x, sheet, paper);
    }
    // alternating rows of gold dust and water: a slurry, not a dry heap
    for y in sheet - 6..sheet {
        for x in 2..w - 2 {
            layer.set_mat(x, y, if y % 2 == 0 { gold } else { water });
        }
    }
    let gold0 = layer.mat.iter().filter(|&&m| m == gold).count();

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 1200);
    let l = &layers[0];

    let count_in = |m: u8, ys: std::ops::Range<u16>| {
        ys.flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| l.mat_at(x, y) == m)
            .count()
    };
    // suspended grains still above the paper count as retained — they have nowhere to go
    let afloat: usize = (0..(sheet as usize) * w as usize)
        .filter(|&i| l.susp_mat[i] == gold)
        .map(|i| l.susp_conc[i] as usize / ENTRAIN_UNITS as usize)
        .sum();
    let gold_above = count_in(gold, 0..sheet) + afloat;
    let gold_below = count_in(gold, sheet + 1..h - 1);
    let water_below = count_in(water, sheet + 1..h - 1);
    println!(
        "of {gold0} gold: {gold_above} retained above the paper, {gold_below} washed through; \
         {water_below} water cells drained"
    );

    assert!(
        water_below > 0,
        "the paper passed no water at all — it is a filter, not a wall"
    );
    assert!(
        gold_below * 10 < gold0,
        "{gold_below} of {gold0} gold cells washed through an 8 um filter"
    );
}

/// A one-cell-thick cardboard shelf under water soaks, sags into a connected curve pinned
/// at its ends, and then stops.
///
/// Two bugs live here, both invisible in a thick-walled cup:
///
/// * **Thin sheets could not get wet at all.** Absorption needs [`ABSORB_UNITS`] of free
///   capacity before it will spend a cell of water, and a single row of board only offers
///   three cells of the old 3x3 patch — 765 against 896 — so it refused for ever and the
///   shelf stayed bone dry for 3000 ticks. Every real box is a thin sheet.
/// * **The sag tore itself to pieces.** Cells fell independently, each on its own hash
///   phase, and the shelf came apart into a descending cloud of loose pixels instead of
///   giving way in the middle.
///
/// So the shape is the assertion, not just the movement: every column that has dropped must
/// stay within one row of its neighbours, which is what makes it a sheet rather than
/// confetti, and the whole thing must come to rest.
#[test]
fn a_soaked_shelf_sags_into_a_curve_and_stops() {
    let t = table();
    let lut = lut();
    let (water, card, rock) = (id(&t, "water"), id(&t, "cardboard"), id(&t, "bedrock"));
    let (w, h) = (40u16, 40u16);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, 11);
    for x in 0..w {
        layer.set_mat(x, h - 1, rock);
    }
    // a shelf one cell thick, built into rock pillars at both ends
    let shelf = 14u16;
    for y in 0..shelf {
        for x in [0u16, 1, w - 2, w - 1] {
            layer.set_mat(x, y, rock);
        }
    }
    for x in 2..w - 2 {
        layer.set_mat(x, shelf, card);
    }
    for y in shelf - 4..shelf {
        for x in 2..w - 2 {
            layer.set_mat(x, y, water);
        }
    }
    let card0 = layer.mat.iter().filter(|&&m| m == card).count();

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 300);
    let soaked = (0..layers[0].mat.len())
        .filter(|&i| layers[0].mat[i] == card)
        .map(|i| layers[0].wetness[i])
        .max()
        .unwrap();
    println!("thin shelf wetness after 300 ticks: {soaked}");
    assert!(
        soaked > 0,
        "a one-cell-thick sheet never absorbed anything — the patch cannot reach far enough \
         to hold a whole cell of water"
    );

    // the lowest cardboard cell per column, for columns that hold any
    let profile = |l: &Layer| -> Vec<Option<u16>> {
        (0..w)
            .map(|x| (0..h).rev().find(|&y| l.mat_at(x, y) == card))
            .collect()
    };
    tick_n(&mut layers, &t, &lut, 1700);
    let mid = profile(&layers[0]);
    tick_n(&mut layers, &t, &lut, 1500);
    let end = profile(&layers[0]);

    let deepest = end.iter().flatten().max().copied().unwrap();
    println!(
        "shelf sagged from row {shelf} to row {deepest}; {} cells, was {card0}",
        layers[0].mat.iter().filter(|&&m| m == card).count()
    );
    assert_eq!(
        layers[0].mat.iter().filter(|&&m| m == card).count(),
        card0,
        "cardboard cells went missing — the shelf must deform, not lose material"
    );
    assert!(deepest > shelf, "the shelf never sagged");
    assert_eq!(mid, end, "the sag never came to rest");

    // No column may sit more than a row away from its neighbour: that is the difference
    // between a sagging sheet and a cloud of loose pixels.
    let cols: Vec<u16> = end.iter().flatten().copied().collect();
    assert_eq!(cols.len(), (w - 4) as usize, "the shelf lost whole columns");
    for (i, pair) in cols.windows(2).enumerate() {
        let step = pair[1].abs_diff(pair[0]);
        assert!(
            step <= 1,
            "the sheet tore: column {} sits at row {} and its neighbour at {}",
            i + 2,
            pair[0],
            pair[1]
        );
    }
    // and it is a sag, not a slide: the ends stay where they were built in
    assert_eq!(cols[0], shelf, "the pinned end let go");
    assert_eq!(*cols.last().unwrap(), shelf, "the pinned end let go");
}

/// A thin stream poured onto a pile has to run *down* the pile, not ski past it.
///
/// The liquid solver had no diagonal step: it fell straight, and failing that dispersed
/// laterally to the furthest cell with somewhere to fall. Nothing in that can follow a
/// surface. A one-cell stream on the apex of a 45-degree pile therefore had each arriving
/// cell leap the full dispersion distance to the same x and then drop straight, side
/// alternating on tick parity — two symmetric dotted columns of mid-air droplets, two rows
/// apart, never touching the sand they were supposedly running over.
///
/// The metric is *lonely* water: a cell with all eight neighbours empty. Stream cells have
/// water above and below, and a rivulet has sand beside it, so only a ballistic droplet in
/// open air is lonely. Summed over 600 ticks to measure the steady state rather than
/// whichever frame the assertion happens to land on.
#[test]
fn a_stream_runs_down_a_pile_instead_of_raining_past_it() {
    let t = table();
    let lut = lut();
    let (water, sand, rock) = (id(&t, "water"), id(&t, "sand"), id(&t, "bedrock"));
    let (w, h) = (80u16, 48u16);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, 3);
    for x in 0..w {
        layer.set_mat(x, h - 1, rock);
    }
    // a 45-degree pile, apex centred under the stream
    for dx in -30i32..=30 {
        for k in 0..(30 - dx.abs()) {
            let (x, y) = (40 + dx, h as i32 - 2 - k);
            if layer.in_bounds(x, y) {
                layer.set_mat(x as u16, y as u16, sand);
            }
        }
    }

    let lonely = |l: &Layer| -> u32 {
        let mut n = 0;
        for y in 0..l.h {
            for x in 0..l.w {
                if l.mat_at(x, y) != water {
                    continue;
                }
                let touching = (-1i32..=1).any(|dy| {
                    (-1i32..=1).any(|dx| {
                        (dx != 0 || dy != 0)
                            && l.in_bounds(x as i32 + dx, y as i32 + dy)
                            && l.mat_at((x as i32 + dx) as u16, (y as i32 + dy) as u16) != 0
                    })
                });
                if !touching {
                    n += 1;
                }
            }
        }
        n
    };

    let mut layers = vec![layer];
    let mut airborne = 0u32;
    for tick in 1..=900u64 {
        if layers[0].mat_at(40, 2) == 0 {
            layers[0].set_mat(40, 2, water); // a one-cell stream
        }
        step_all_serial(&mut layers, &[], &t, &lut, tick);
        if tick > 300 {
            airborne += lonely(&layers[0]);
        }
    }
    println!("lonely droplet-ticks over 600 ticks: {airborne}");
    // Measured 5718 with the diagonal step removed, 2160 with it. The remainder is splash
    // at the shoreline, where dispersion still throws the odd cell off the water's edge —
    // irregular, unlike the metronome this test is about, and it reads as splash.
    // Gating dispersion on "has a liquid neighbour" was tried as a way to cut it further
    // and measured *worse* (6449): starving the apex of dispersion piles cells up there
    // and the collapse throws more water than it saves.
    assert!(
        airborne < 3500,
        "the stream is raining past the pile rather than running down it: \
         {airborne} lonely droplet-ticks"
    );
}

/// The entrain/deposit cycle used to be an unbounded water sink, and this is the audit
/// that catches it: both budgets, counted in the units each is stored in.
///
/// Water lives in three places at once and all three have to be summed, or the test passes
/// while the water quietly changes form: whole liquid cells, wetness inside powder, and the
/// pore-water *film* a suspended grain carries in its carrier's wetness channel. One liquid
/// cell is worth `ABSORB_UNITS` of wetness, which is the exchange rate absorption uses.
///
/// Fine powder is the case that matters — grain size decides whether a powder enters the
/// suspension path at all, so sand (400 µm) never did and was never affected, which is
/// exactly what made the bug present as "fine powders absorb too much".
#[test]
fn a_pool_over_fine_powder_conserves_both_budgets() {
    let t = table();
    let lut = lut();
    let (water, rock) = (id(&t, "water"), id(&t, "bedrock"));

    for powder_name in ["silt", "gold_dust", "sand"] {
        let powder = id(&t, powder_name);
        let (w, h) = (96u16, 96u16);
        let mut layer = Layer::new(w, h, LayerSlot::Plant, 7);
        for x in 0..w {
            layer.set_mat(x, h - 1, rock);
            layer.set_mat(x, h - 2, rock);
        }
        for y in 0..h {
            for x in [0, 1, w - 2, w - 1] {
                layer.set_mat(x, y, rock);
            }
        }
        // one brush of powder on the floor
        for dy in -6i32..=6 {
            for dx in -6i32..=6 {
                if dx * dx + dy * dy > 36 {
                    continue;
                }
                let (x, y) = (48 + dx, (h - 3) as i32 + dy);
                if layer.in_bounds(x, y) && layer.mat_at(x as u16, y as u16) == 0 {
                    layer.set_mat(x as u16, y as u16, powder);
                }
            }
        }
        // and a pool poured over it
        for y in (h - 23)..(h - 3) {
            for x in 2..(w - 2) {
                if layer.mat_at(x, y) == 0 {
                    layer.set_mat(x, y, water);
                }
            }
        }

        // wetness units, counting a whole liquid cell as ABSORB_UNITS of them
        let water_units = |l: &Layer| -> u64 {
            let cells = l.mat.iter().filter(|&&m| m == water).count() as u64;
            let held: u64 = l.wetness.iter().map(|&v| v as u64).sum();
            cells * ABSORB_UNITS as u64 + held
        };
        // cells of powder, counting a suspended grain as the cell it will deposit as
        let powder_cells = |l: &Layer| -> u64 {
            let settled = l.mat.iter().filter(|&&m| m == powder).count() as u64;
            let afloat: u64 = (0..l.mat.len())
                .filter(|&i| l.susp_mat[i] == powder)
                .map(|i| l.susp_conc[i] as u64)
                .sum();
            settled + afloat / ENTRAIN_UNITS as u64
        };

        let mut layers = vec![layer];
        let (w0, p0) = (water_units(&layers[0]), powder_cells(&layers[0]));
        tick_n(&mut layers, &t, &lut, 1600);
        let (w1, p1) = (water_units(&layers[0]), powder_cells(&layers[0]));
        println!(
            "{powder_name}: water {w0} -> {w1} units ({:+.1} cells), powder {p0} -> {p1} cells",
            (w1 as f64 - w0 as f64) / ABSORB_UNITS as f64
        );

        // Evaporation is a real sink and the pool has a free surface, so allow a few cells of
        // slack. The bug this guards was 99% of the pool, and still 17% after the first half
        // of the fix — nothing near this bound.
        let lost = w0.saturating_sub(w1);
        assert!(
            lost < 8 * ABSORB_UNITS as u64,
            "{powder_name}: {:.0} cells of water destroyed ({lost} of {w0} units)",
            lost as f64 / ABSORB_UNITS as f64
        );
        assert!(w1 <= w0, "{powder_name}: water was created out of nothing");
        assert_eq!(p0, p1, "{powder_name}: powder cells were not conserved");
    }
}

#[test]
fn undo_restores_the_players_cells_but_not_the_sims() {
    let t = table();
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 1);
    let stone = t.id("stone").unwrap();
    let mut journal = UndoJournal::new();
    assert_eq!(journal.depth(), 0);

    let rect = apply_rect(
        &mut layer,
        &mut journal,
        0,
        &t,
        CellRect {
            x0: 10,
            y0: 10,
            x1: 20,
            y1: 20,
        },
        true,
        Tool::Paint(stone),
        Mirror::default(),
    );
    assert_eq!(rect.width(), 11);
    assert_eq!(layer.mat_at(15, 15), stone.0);
    assert_eq!(journal.depth(), 1);

    // the sim (or anything else) overwrites one cell and drops the ownership bit
    let i = layer.idx(15, 15);
    layer.mat[i] = t.id("sand").unwrap().0;
    layer.flags[i] &= !FLAG_PLAYER_PLACED;

    let mut layers = vec![layer];
    assert!(journal.undo(&mut layers));
    assert_eq!(journal.depth(), 0);
    assert_eq!(layers[0].mat_at(10, 10), 0, "player cell not rolled back");
    assert_eq!(layers[0].mat_at(20, 20), 0, "player cell not rolled back");
    assert_eq!(
        layers[0].mat_at(15, 15),
        t.id("sand").unwrap().0,
        "undo resurrected sim history"
    );
    assert!(!journal.undo(&mut layers), "undo popped an empty journal");
}

#[test]
fn mirror_and_clipboard_round_trip() {
    let t = table();
    let stone = t.id("stone").unwrap();
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 1);
    let mut journal = UndoJournal::new();

    apply_brush(
        &mut layer,
        &mut journal,
        0,
        &t,
        (10, 10),
        Brush::Square { r: 1 },
        Tool::Paint(stone),
        Mirror {
            x: Some(32),
            y: None,
        },
    );
    // mirrored about x = 32: 10 -> 54
    assert_eq!(layer.mat_at(10, 10), stone.0);
    assert_eq!(layer.mat_at(54, 10), stone.0);

    let clip = copy_region(
        &layer,
        CellRect {
            x0: 9,
            y0: 9,
            x1: 11,
            y1: 11,
        },
    );
    assert_eq!((clip.w, clip.h), (3, 3));
    assert_eq!(clip.aux.len(), 9 * 7);
    paste_region(
        &mut layer,
        &mut journal,
        0,
        &clip,
        (40, 40),
        Mirror::default(),
    );
    assert_eq!(layer.mat_at(41, 41), stone.0);
}

#[test]
fn dig_refuses_bedrock_and_flood_fill_is_bounded() {
    let t = table();
    let mut layer = Layer::new(256, 256, LayerSlot::Plant, 1);
    let mut journal = UndoJournal::new();
    let bedrock = t.id("bedrock").unwrap();
    let stone = t.id("stone").unwrap();

    layer.fill(bedrock.0);
    apply_brush(
        &mut layer,
        &mut journal,
        0,
        &t,
        (100, 100),
        Brush::Square { r: 3 },
        Tool::Dig,
        Mirror::default(),
    );
    assert_eq!(layer.mat_at(100, 100), bedrock.0, "bedrock was dug");

    layer.fill(stone.0);
    apply_brush(
        &mut layer,
        &mut journal,
        0,
        &t,
        (100, 100),
        Brush::Square { r: 3 },
        Tool::Dig,
        Mirror::default(),
    );
    assert_eq!(layer.mat_at(100, 100), 0, "stone was not dug");

    // flood fill stops at the 128x128 window around the seed
    layer.fill(stone.0);
    let rect = flood_fill(&mut layer, &mut journal, 0, &t, (128, 128), bedrock);
    assert_eq!(rect.width(), 129);
    assert_eq!(layer.mat_at(128, 128), bedrock.0);
    assert_eq!(layer.mat_at(0, 0), stone.0, "flood fill escaped its window");
    assert_eq!(
        layer.mat_at(255, 255),
        stone.0,
        "flood fill escaped its window"
    );
}

#[test]
fn cell_rect_and_bitset_behave() {
    let r = CellRect {
        x0: 20,
        y0: 5,
        x1: 3,
        y1: 90,
    }
    .clamped(64, 64);
    assert_eq!((r.x0, r.x1, r.y0, r.y1), (3, 20, 5, 63));
    assert_eq!(r.width(), 18);
    assert_eq!(r.height(), 59);
    assert!(r.contains(10, 10));
    assert!(!r.contains(2, 10));

    let mut b = Bitset::new(200);
    assert_eq!(b.count(), 0);
    b.set(0);
    b.set(63);
    b.set(64);
    b.set(199);
    assert_eq!(b.count(), 4);
    assert_eq!(b.iter_set().collect::<Vec<_>>(), vec![0, 63, 64, 199]);
    b.clear(64);
    assert!(!b.get(64));
    b.clear_all();
    assert_eq!(b.count(), 0);
}

#[test]
fn vessel_frame_maps_world_to_cells() {
    let mut f = VesselFrame::new(64, 32);
    assert_eq!(f.world_to_cell([0.5, 0.5]), Some((0, 0)));
    assert_eq!(f.world_to_cell([63.9, 31.9]), Some((63, 31)));
    assert_eq!(f.world_to_cell([-0.1, 0.0]), None);
    assert_eq!(f.world_to_cell([64.0, 0.0]), None);
    f.origin = [100.0, 200.0];
    assert_eq!(f.world_to_cell([100.0, 200.0]), Some((0, 0)));
    assert_eq!(f.cell_to_world((0, 0)), [100.5, 200.5]);
}

#[test]
fn coarse_grid_records_and_decays_flow() {
    let t = table();
    let lut = lut();
    let water = id(&t, "water");
    let mut layer = Layer::new(64, 64, LayerSlot::Plant, 1);
    for x in 20..44u16 {
        layer.set_mat(x, 0, water);
    }
    assert_eq!(layer.coarse.cw, 8);
    assert_eq!(layer.coarse.ch, 8);

    let mut layers = vec![layer];
    tick_n(&mut layers, &t, &lut, 3);
    let flow: u32 = layers[0].coarse.flow_mag.iter().map(|&v| v as u32).sum();
    assert!(flow > 0, "falling water recorded no coarse flow");

    // heat and pressure are allocated and never touched in the slice
    assert!(layers[0].coarse.heat.iter().all(|&v| v == 0.0));
    assert!(layers[0].coarse.pressure.iter().all(|&v| v == 0.0));

    tick_n(&mut layers, &t, &lut, 400);
    let flow_later: u32 = layers[0].coarse.flow_mag.iter().map(|&v| v as u32).sum();
    assert!(flow_later < flow, "coarse flow never decayed");
}

#[test]
fn stats_report_the_work_actually_done() {
    let t = table();
    let lut = lut();
    let sand = id(&t, "sand");
    let mut layer = Layer::new(128, 128, LayerSlot::Plant, 1);
    for x in 10..40u16 {
        layer.set_mat(x, 10, sand);
    }
    let mut layers = vec![layer];
    let s = step_all_serial(&mut layers, &[], &t, &lut, 1);
    assert_eq!(s.active_cells, 30, "every grain should have been visited");
    assert_eq!(s.moves, 30, "every grain should have fallen");
    assert_eq!(s.per_layer_ms.len(), 1);
    assert!(s.awake_chunks > 0);
    assert_eq!(s.transfers, 0);
}

// ---- the Bevy plugin and throughput ----------------------------------------

#[test]
fn the_plugin_builds_and_steps_a_real_app() {
    use bevy::app::ScheduleRunnerPlugin;
    use bevy::diagnostic::{DiagnosticsPlugin, DiagnosticsStore};
    use bevy::prelude::*;

    let mut app = App::new();
    app.add_plugins((
        MinimalPlugins.set(ScheduleRunnerPlugin::run_once()),
        DiagnosticsPlugin,
    ))
    .add_plugins(PixelSimPlugin {
        config: SimConfig {
            width: 256,
            height: 128,
            slots: vec![LayerSlot::Face, LayerSlot::Plant, LayerSlot::Gangway],
            seed: 0x5EED_0002,
            material_path: None,
            // the startup bisection is a second of CPU; exercised by its own test
            calibrate_repose: false,
            parallel: true,
        },
    });

    // Startup: load_materials then build_layers
    app.finish();
    app.update();

    {
        let world = app.world_mut();
        assert!(world.get_resource::<Materials>().is_some(), "no Materials");
        assert!(world.get_resource::<Repose>().is_some(), "no Repose");
        let sand = world.resource::<Materials>().id("sand").unwrap().0;
        let mut sim = world.resource_mut::<Sim>();
        assert_eq!(sim.layers.len(), 3);
        // Face, Plant, Gangway: the Plant is the middle slab and the only simulated one
        assert_eq!(sim.index_of(LayerSlot::Plant), Some(1));
        assert_eq!(sim.depth_from(1, 2), 1);
        assert_eq!(sim.depth_from(1, 0), -1);
        // a hopper in the Plant...
        for x in 100..140u16 {
            sim.layer_mut(1).set_mat(x, 4, sand);
        }
        // ...and the same hopper in a *static* slab, which must still be there at the end.
        // This is `One Simulated Plane` as a behavioural assertion rather than a comment:
        // the Gangway is structure, and structure does not fall.
        for x in 100..140u16 {
            sim.layer_mut(2).set_mat(x, 4, sand);
        }
    }

    // `Time<Fixed>` is fed by the real clock, and 400 back-to-back `update()` calls
    // take far less than 400/60 s of wall time, so drive the schedule the plugin
    // registered its systems into directly.
    for _ in 0..400 {
        app.world_mut().run_schedule(FixedUpdate);
    }

    let world = app.world();
    let sand = world.resource::<Materials>().id("sand").unwrap().0;
    let stats = world.resource::<Stats>();
    assert!(stats.0.step_ms >= 0.0);
    assert_eq!(stats.0.per_layer_ms.len(), 3);
    let sim = world.resource::<Sim>();
    assert!(
        sim.layer(1).tick > 0,
        "FixedUpdate never ran a sim tick (tick = {})",
        sim.layer(1).tick
    );
    // the sand fell out of the hopper and is now in the bottom eighth of the layer
    assert_eq!(sim.layer(1).mat_at(120, 4), 0);
    let deck_rows: Vec<u16> = (0..128u16)
        .filter(|&y| (0..256u16).any(|x| sim.layer(1).mat_at(x, y) != 0))
        .collect();
    assert!(!deck_rows.is_empty(), "the sand vanished from the Plant");
    assert!(
        deck_rows[0] >= 112,
        "sand is still high up the Plant, topmost row {}",
        deck_rows[0]
    );
    // The static slab's sand has not moved a cell in 400 ticks.
    for x in 100..140u16 {
        assert_eq!(
            sim.layer(2).mat_at(x, 4),
            sand,
            "sand moved in a static slab at x={x}: the Gangway is being simulated"
        );
    }
    assert!(
        sim.layer(2).stats.moves == 0 && sim.layer(2).stats.awake_chunks == 0,
        "a static slab reported work: {} moves, {} awake chunks",
        sim.layer(2).stats.moves,
        sim.layer(2).stats.awake_chunks
    );
    // and it costs nothing to not simulate it
    let per_layer = &world.resource::<Stats>().0.per_layer_ms;
    assert!(
        per_layer[2] <= per_layer[1],
        "static slab {:.3} ms cost more than the Plant {:.3} ms",
        per_layer[2],
        per_layer[1]
    );

    // the diagnostics were registered and are being fed
    let store = world.resource::<DiagnosticsStore>();
    for path in [&DIAG_STEP_MS, &DIAG_ACTIVE_CELLS, &DIAG_AWAKE_CHUNKS] {
        assert!(store.get(path).is_some(), "diagnostic {path:?} missing");
    }

    // pause gates the step
    let tick_before = sim.layer(0).tick;
    app.world_mut().resource_mut::<Paused>().0 = true;
    for _ in 0..60 {
        app.world_mut().run_schedule(FixedUpdate);
    }
    assert_eq!(app.world().resource::<Sim>().layer(0).tick, tick_before);
    app.world_mut().resource_mut::<StepOnce>().0 = true;
    for _ in 0..60 {
        app.world_mut().run_schedule(FixedUpdate);
    }
    assert_eq!(
        app.world().resource::<Sim>().layer(0).tick,
        tick_before + 1,
        "StepOnce advanced more than a single tick"
    );
}

/// Not a benchmark, an upper bound: three 1024x512 layers with tens of thousands of
/// moving cells must step well inside a 16.6 ms frame. Run with `--release`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "throughput is only meaningful in --release"
)]
fn a_busy_three_layer_vessel_steps_inside_a_frame() {
    let t = table();
    let lut = lut();
    let sand = id(&t, "sand");
    let water = id(&t, "water");

    let mut layers: Vec<Layer> = [LayerSlot::Face, LayerSlot::Plant, LayerSlot::Gangway]
        .into_iter()
        .enumerate()
        .map(|(i, slot)| Layer::new(1024, 512, slot, 0xBEEF_0000 + i as u64))
        .collect();
    // ~110k cells of loose material per layer, dropped from the top third.
    //
    // The mix and the depth are both load knobs, and both have had to go up as the wet and
    // liquid rules got better rather than the bound coming down. Two thirds loose rather
    // than one half, because standing water now soaks into dry sand instead of sitting on
    // it, so a 50/50 mix turns much of its water into wetness in the first few ticks. And
    // 176 rows rather than 160, because the liquid solver's diagonal step lets water reach
    // its rest state sooner, which is the point of it — a settled cell is not an active one.
    for (li, layer) in layers.iter_mut().enumerate() {
        for y in 0..176u16 {
            for x in 0..1024u16 {
                let h = hash_rng(0x51EE + li as u64, 0, x, y);
                match h % 3 {
                    0 => layer.set_mat(x, y, sand),
                    1 => layer.set_mat(x, y, water),
                    _ => {}
                }
            }
        }
    }

    let mut ticks = Vec::with_capacity(240);
    let mut peak_active = 0u32;
    let mut total = 0.0f32;
    for tick in 1..=240u64 {
        let s = step_all_serial(&mut layers, &[], &t, &lut, tick);
        ticks.push(s.step_ms);
        peak_active = peak_active.max(s.active_cells);
        total += s.step_ms;
    }
    // Judge the median, not the worst. `cargo test` runs this concurrently with the rest
    // of the suite, so a single tick can absorb an arbitrary amount of somebody else's
    // load — a worst-tick threshold measures the host, not the solver.
    let serial_med = median(&mut ticks);
    println!(
        "3x 1024x512 serial:   peak active cells {peak_active}, median tick {serial_med:.2} ms, \
         mean {:.2} ms",
        total / 240.0
    );
    assert!(peak_active > 100_000, "scene was not busy: {peak_active}");

    // the same scene through the task pool, which is how the plugin runs it
    let pool = bevy::tasks::TaskPool::new();
    let mut layers: Vec<Layer> = [LayerSlot::Face, LayerSlot::Plant, LayerSlot::Gangway]
        .into_iter()
        .enumerate()
        .map(|(i, slot)| Layer::new(1024, 512, slot, 0xBEEF_0000 + i as u64))
        .collect();
    for (li, layer) in layers.iter_mut().enumerate() {
        for y in 0..160u16 {
            for x in 0..1024u16 {
                match hash_rng(0x51EE + li as u64, 0, x, y) % 4 {
                    0 => layer.set_mat(x, y, sand),
                    1 => layer.set_mat(x, y, water),
                    _ => {}
                }
            }
        }
    }
    let mut p_ticks = Vec::with_capacity(240);
    let mut p_total = 0.0f32;
    for tick in 1..=240u64 {
        let s = step_all(&mut layers, &[], &t, &lut, tick, &pool);
        p_ticks.push(s.step_ms);
        p_total += s.step_ms;
    }
    let pooled_med = median(&mut p_ticks);
    println!(
        "3x 1024x512 pooled:   median tick {pooled_med:.2} ms, mean {:.2} ms \
         ({} pool threads)",
        p_total / 240.0,
        pool.thread_num()
    );
    // Serial is one thread doing all three layers, so it is allowed to blow the frame;
    // it is here as the ceiling. The pooled path is how the plugin actually runs, and
    // measures ~9 ms for this scene on an M1 — a 3x wall means real regressions trip it
    // without the suite's own load doing so.
    assert!(
        serial_med < 60.0,
        "median serial tick {serial_med:.2} ms is beyond any plausible frame budget"
    );
    assert!(
        pooled_med < 30.0,
        "median pooled tick {pooled_med:.2} ms; this scene steps in ~9 ms on an M1"
    );
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// The full-fidelity calibration the plugin runs at startup, at the plan's grain
/// count. Slow (seconds), so it is not part of the default run.
#[test]
#[ignore = "slow: 4000-grain piles for every authored angle"]
fn calibration_hits_every_authored_angle_at_full_fidelity() {
    let t = table();
    let angles: Vec<f32> = t
        .iter()
        .filter(|(_, m)| m.class == MaterialClass::Powder)
        .map(|(_, m)| m.repose_angle)
        .collect();
    let mut lut = ReposeLut::analytic();
    let t0 = std::time::Instant::now();
    lut.calibrate(&t, &angles, 4000, 0x5EED_1234_ABCD_0001);
    println!(
        "calibrated {} angles in {:.0} ms",
        angles.len(),
        t0.elapsed().as_secs_f32() * 1000.0
    );

    for (id, m) in t.iter() {
        if m.class != MaterialClass::Powder {
            continue;
        }
        let measured = ReposeLut::measure_pile_angle(&drop_pile(&t, &lut, id.0, 4000, 99));
        println!(
            "{:<10} authored {:.1} deg, measured {measured:.2} deg (span {} drop {})",
            m.name,
            m.repose_angle,
            lut.creep_span(m.repose_angle),
            lut.creep_drop(m.repose_angle)
        );
        assert!(
            (measured - m.repose_angle).abs() <= 2.5,
            "{} piled at {measured:.2} deg, authored {:.1}",
            m.name,
            m.repose_angle
        );
    }
}
