//! Adversarial-review reproductions. Each of these documented a defect and failed against
//! the behaviour of the day; they are kept as the regression tests for those fixes and now
//! run in the normal suite, unignored and with their assertions untouched.
//!
//! The four that assert a wall-clock budget take `timed()` so they never measure each
//! other's load, and carry pixelsim's
//! `#[cfg_attr(debug_assertions, ignore)]`, for the same reason it does: `cargo test`
//! defaults to debug, where every number here is 30-50x larger for reasons that have
//! nothing to do with the code under test. Their thresholds are untouched.
//!
//! `blur_alone` is one of them: it was a print-only harness, and now guards the separable
//! blur against the 8.30 ms/call the 9-tap version cost. For the raw numbers:
//!   cargo test -p pixelview --test zz_review --release -- --nocapture
use bevy::asset::Assets;
use bevy::image::Image;
use pixelsim::*;
use pixelview::*;

/// The wall-clock tests below measure sub-0.05 ms budgets, so they cannot share the
/// machine with each other — `blur_alone` alone is enough to push the one-tile upload
/// over its threshold. Every timed test takes this first.
fn timed() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn tab() -> MaterialTable {
    MaterialTable::embedded()
}

fn scene(w: u16, h: u16) -> Layer {
    let t = tab();
    let sand = t.id("sand").unwrap().0;
    let mut l = Layer::new(w, h, LayerSlot::Plant, 1);
    for y in h / 2..h {
        for x in 0..w {
            let i = l.idx(x, y);
            l.mat[i] = sand;
        }
    }
    l
}

/// Toggling x-ray changes every pixel of the ACTIVE layer. It used to leave every tile
/// the sim had not happened to dirty showing the un-ghosted bake, because nothing set
/// `force_full`. `bake_with` now fingerprints the treatment and rebakes whole itself.
#[test]
fn xray_toggle_rebakes_the_active_layer() {
    let t = tab();
    let mut images = Assets::<Image>::default();
    let pal = Palette::from_table(&t);
    let cfg = RenderConfig::default();
    let mut layer = scene(128, 128);

    let mut canvas = LayerCanvas::new(&mut images, 128, 128, false, true);
    // first bake: full, no x-ray
    canvas.bake_with(
        &mut layer,
        &t,
        &pal,
        &cfg.treatment_for(0, 0, LayerSlot::Plant, false),
        false,
        false,
    );
    let plain = canvas.rgba.clone();

    // player presses X. depth 0 under x-ray: same half_res, same blur, so the bake
    // gate is the only thing that runs -- and there is nothing dirty.
    let xr = cfg.treatment_for(0, 0, LayerSlot::Plant, true);
    assert_eq!(xr.half_res, false);
    assert_eq!(xr.blur_px, 0.0);
    canvas.bake_with(&mut layer, &t, &pal, &xr, false, true);
    let ghosted = canvas.rgba.clone();

    // what it SHOULD look like
    canvas.force_full = true;
    canvas.bake_with(&mut layer, &t, &pal, &xr, false, true);
    let expected = canvas.rgba.clone();

    let stale = ghosted
        .chunks(4)
        .zip(expected.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "xray: {stale} of {} pixels are stale after toggling x-ray (plain==ghosted: {})",
        128 * 128,
        plain == ghosted
    );
    assert_eq!(stale, 0, "x-ray toggle left {stale} stale pixels");
}

/// The schematic used to be rebuilt from a full class histogram of every cell of every
/// layer on every tick the zoom crossfade was up. It now rescans only the chunks whose
/// cells actually moved, decided by a per-chunk content hash.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "throughput is only meaningful in --release"
)]
fn schematic_bake_cost() {
    let _timed = timed();
    let t = tab();
    let mut images = Assets::<Image>::default();
    let layers: Vec<Layer> = (0..3).map(|_| scene(1024, 512)).collect();
    let mut canvas = LayerCanvas::new(
        &mut images,
        1024u16.div_ceil(CELL_PX),
        512u16.div_ceil(CELL_PX),
        false,
        true,
    );

    let t0 = std::time::Instant::now();
    let n = 60;
    for _ in 0..n {
        canvas.clear();
        for (i, l) in layers.iter().enumerate().rev() {
            bake_schematic(&mut canvas, l, &t, i as i32);
        }
    }
    let per_tick_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("schematic composite: {per_tick_ms:.2} ms/tick for 3x1024x512 (budget is 16.6)");
    assert!(
        per_tick_ms < 1.0,
        "schematic composite costs {per_tick_ms:.2} ms/tick"
    );
}

/// Upload used to copy the whole texture even when one 64x64 tile changed. It now copies
/// only the rows the bake touched.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "throughput is only meaningful in --release"
)]
fn upload_cost_is_proportional_to_the_dirty_area() {
    let _timed = timed();
    let t = tab();
    let mut images = Assets::<Image>::default();
    let pal = Palette::from_table(&t);
    let cfg = RenderConfig::default();
    let mut layer = scene(1024, 512);
    let mut canvas = LayerCanvas::new(&mut images, 1024, 512, false, true);
    let tr = cfg.treatment_for(0, 0, LayerSlot::Plant, false);
    canvas.bake_with(&mut layer, &t, &pal, &tr, false, false);
    canvas.upload(&mut images);

    // dirty exactly one cell
    layer.set_mat(500, 300, t.id("water").unwrap().0);
    let t0 = std::time::Instant::now();
    let n = 200;
    for _ in 0..n {
        layer.chunks.mark_dirty(500, 300);
        canvas.bake_with(&mut layer, &t, &pal, &tr, false, false);
        canvas.upload(&mut images);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("one dirty 64x64 tile -> bake+upload = {ms:.3} ms/tick (4096 of 524288 px changed)");
    assert!(ms < 0.05, "one-tile update costs {ms:.3} ms");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "throughput is only meaningful in --release"
)]
fn split_bake_and_upload() {
    let _timed = timed();
    let t = tab();
    let mut images = Assets::<Image>::default();
    let pal = Palette::from_table(&t);
    let cfg = RenderConfig::default();
    let mut layer = scene(1024, 512);
    let mut canvas = LayerCanvas::new(&mut images, 1024, 512, false, true);
    let tr = cfg.treatment_for(0, 0, LayerSlot::Plant, false);
    canvas.bake_with(&mut layer, &t, &pal, &tr, false, false);
    canvas.upload(&mut images);
    let n = 500;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        layer.chunks.mark_dirty(500, 300);
        canvas.bake_with(&mut layer, &t, &pal, &tr, false, false);
    }
    let bake = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        canvas.upload(&mut images);
    }
    let up = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!(
        "one-tile bake = {bake:.4} ms   full-buffer upload = {up:.4} ms  (ratio {:.0}x)",
        up / bake
    );

    // and a blurred rear layer, which used to rebake and re-blur whole on every call
    let rear = cfg.treatment_for(5, 2, LayerSlot::Gangway, false);
    println!(
        "rear treatment blur_px={} half_res={} bake_hz={}",
        rear.blur_px, rear.half_res, rear.bake_hz
    );
    let t0 = std::time::Instant::now();
    for _ in 0..50 {
        canvas.bake_with(&mut layer, &t, &pal, &rear, false, false);
    }
    let rb = t0.elapsed().as_secs_f64() * 1000.0 / 50.0;
    println!("blurred rear-layer bake, nothing dirty = {rb:.2} ms");
    assert!(rb < 1.0, "rear bake {rb:.2} ms");
}

/// The blur in isolation. It was 8.30 ms/call as a 9-tap with a full buffer clone
/// per call; separable + hoisted scratch has to keep it far under that, because a
/// rear layer pays it every bake.
#[test]
#[cfg_attr(debug_assertions, ignore = "wall-clock; run in release")]
fn blur_alone() {
    let _timed = timed();
    let mut rgba = vec![7u8; 1024 * 512 * 4];
    let n = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        box_blur_rgba(&mut rgba, 1024, 512);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    println!("box_blur_rgba on 1024x512 = {ms:.2} ms/call");
    // ~2.4 ns/px on an M1: memory-bound, so the split buys 1.6x and not the 4.5x the tap
    // count suggests. The real saving is `split_bake_and_upload`'s rear bake, which now
    // skips the blur entirely when nothing is dirty (12.36 ms -> 0.12 ms).
    assert!(ms < 6.5, "blur {ms:.2} ms/call, was 8.30 ms before the split");
}

// ---------------------------------------------------------------- perspective wiring

/// The unit tests prove the perspective *maths*. This proves the *wiring*: that
/// `update_view_geometry` actually runs against real resources and produces distinct
/// per-plane scales, with the active plane at exactly 1.0.
///
/// Worth its own test because every way this can be broken — the system not registered, it
/// running after its readers, `ActiveLayer` read from the wrong place, an empty `ratios`
/// silently falling back to a flat stack via `ViewGeometry::ratio` — compiles fine and
/// shows up only as "the layers look flat", which is exactly the bug this work set out to
/// fix.
#[test]
fn the_geometry_system_produces_a_real_perspective_stack() {
    use bevy::ecs::system::IntoSystem;
    use bevy::prelude::*;

    let slots = [
        LayerSlot::Plant,
        LayerSlot::Cavity,
        LayerSlot::Plant,
        LayerSlot::Gangway,
    ];
    let mut world = World::new();
    world.insert_resource(Sim {
        layers: slots
            .iter()
            .enumerate()
            .map(|(i, &s)| Layer::new(64, 64, s, i as u64))
            .collect(),
        ports: Vec::new(),
        gas: GasPlane::new(64, 64),
    });
    // the player is in Works, two slots back — so planes exist both in front and behind
    world.insert_resource(ActiveLayer(2));
    world.insert_resource(Zoom::default());
    world.insert_resource(Perspective::default());
    world.insert_resource(ViewGeometry::default());
    // no window entity: the system must fall back to a default viewport rather than bail,
    // or the whole stack silently renders flat in any headless context
    let mut system = IntoSystem::into_system(update_view_geometry);
    system.initialize(&mut world);
    system.run((), &mut world);

    let geom = world.resource::<ViewGeometry>();
    println!(
        "wiring: distance {:.1} px, ratios {:?}",
        geom.distance, geom.ratios
    );
    assert_eq!(geom.ratios.len(), slots.len(), "every layer needs a scale");
    assert!(geom.distance > 0.0, "camera distance was never computed");

    // the active plane is the reference and must be pixel-exact
    assert_eq!(geom.ratio(2), 1.0);
    // strictly receding: each plane behind is smaller than the one in front of it
    for i in 1..slots.len() {
        assert!(
            geom.ratio(i) < geom.ratio(i - 1),
            "plane {i} ({}) is not smaller than plane {} ({})",
            geom.ratio(i),
            i - 1,
            geom.ratio(i - 1)
        );
    }
    // planes in front of the player are larger than 1.0, or the stack reads inside-out
    assert!(geom.ratio(0) > 1.0, "front plane {} is not larger", geom.ratio(0));
    // and the spread is big enough to actually see
    let spread = geom.ratio(0) - geom.ratio(slots.len() - 1);
    assert!(spread > 0.05, "the whole stack spans only {spread:.3} of scale");

    // gaps are measured in sim px, from the contiguous slab stack
    assert!((geom.gap(0, 2) - 2.0 * SLAB_DEPTH_PX).abs() < 1e-3, "gap was {}", geom.gap(0, 2));
}

/// Turning perspective off has to give back exactly the flat stack, so the spike's `F1`
/// A/B is a real control rather than a slightly different perspective.
#[test]
fn perspective_off_is_exactly_flat() {
    use bevy::ecs::system::IntoSystem;
    use bevy::prelude::*;

    let mut world = World::new();
    world.insert_resource(Sim {
        layers: (0..3)
            .map(|i| Layer::new(32, 32, LayerSlot::Plant, i))
            .collect(),
        ports: Vec::new(),
        gas: GasPlane::new(32, 32),
    });
    world.insert_resource(ActiveLayer(0));
    world.insert_resource(Zoom::default());
    world.insert_resource(Perspective {
        enabled: false,
        ..Perspective::default()
    });
    world.insert_resource(ViewGeometry::default());
    let mut system = IntoSystem::into_system(update_view_geometry);
    system.initialize(&mut world);
    system.run((), &mut world);

    let geom = world.resource::<ViewGeometry>();
    // Not exactly 1.0: the slabs collapse to `FLAT_DEPTH_PX` rather than to zero, because
    // four coplanar slabs z-fight and a flickering stack is not a useful control. Half a sim
    // px of separation is flat to the eye: 0.2% of apparent scale per slab at a typical
    // camera distance, against the 11% a real slab of depth produces.
    for i in 0..3 {
        assert!(
            (geom.ratio(i) - 1.0).abs() < 0.01,
            "plane {i} was not flat: {}",
            geom.ratio(i)
        );
    }
    assert!(geom.plane_z[2] > -2.0, "the stack did not collapse: {:?}", geom.plane_z);
}

/// Slab depth is a live knob, and `depth_scale` is how it is turned. It used to multiply a
/// per-slot depth table; when the table went away the multiply went with it, so `F5`/`F6`
/// logged a new number and moved nothing — a knob wired to a log line.
#[test]
fn depth_scale_actually_changes_the_z_step() {
    let base = Perspective::default();
    let half = Perspective { depth_scale: 0.5, ..base };
    let deep = Perspective { depth_scale: 2.0, ..base };
    assert!(
        (half.depth() - base.depth() * 0.5).abs() < 1e-4,
        "halving the scale gave {} against {}",
        half.depth(),
        base.depth()
    );
    assert!(deep.depth() > base.depth(), "exaggerating the stack did nothing");
    // and it may never collapse the stack into a z-fight, however far down it is turned
    let flat = Perspective { depth_scale: 0.0, ..base };
    assert!(flat.depth() >= FLAT_DEPTH_PX, "zero scale produced coplanar slabs");
    // perspective off still wins: the A/B is a control, not a suggestion
    let off = Perspective { enabled: false, depth_scale: 4.0, ..base };
    assert_eq!(off.depth(), FLAT_DEPTH_PX);
}

/// Changing the depth moves every vertex of every slab, not only the chunks the sim
/// dirtied — so the remesh cannot be driven by dirty rects alone. With a static scene and no
/// dirt at all, turning the knob has to still rebuild the stack, or the geometry keeps its
/// old depth while the camera and cursor move to the new one.
#[test]
fn changing_the_depth_remeshes_slabs_the_sim_never_dirtied() {
    use bevy::prelude::*;

    let mut world = World::new();
    let mut images = Assets::<Image>::default();
    let mut meshes = Assets::<Mesh>::default();

    let layers: Vec<Layer> = (0..2).map(|_| scene(64, 64)).collect();
    // Fresh canvases have never baked, so they report no caster rects: nothing is dirty and
    // the only reason to remesh is the depth itself.
    let canvases: Vec<LayerCanvas> = layers
        .iter()
        .map(|l| LayerCanvas::new(&mut images, l.w, l.h, false, true))
        .collect();
    let schematic = LayerCanvas::new(&mut images, 64, 64, false, true);

    let perspective = Perspective::default();
    let built = perspective.depth();
    let mut handles = Vec::new();
    for (i, layer) in layers.iter().enumerate() {
        let (front, back) = slab_z(i, built);
        for chunk in 0..layer.chunks.len() {
            let mut buf = MeshBuf::default();
            build_front(&mut buf, layer, chunk, front);
            build_walls(&mut buf, layer, chunk, front, back);
            let h = meshes.add(buf.into_mesh());
            handles.push(h.clone());
            world.spawn((Mesh3d(h), SlabChunk { layer: i, chunk }));
        }
    }

    // how far the geometry reaches in z, over the whole stack
    let z_span = |meshes: &Assets<Mesh>, handles: &[Handle<Mesh>]| -> f32 {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for h in handles {
            let Some(mesh) = meshes.get(h) else { continue };
            let Some(pos) = mesh.attribute(Mesh::ATTRIBUTE_POSITION) else {
                continue;
            };
            for p in pos.as_float3().unwrap() {
                lo = lo.min(p[2]);
                hi = hi.max(p[2]);
            }
        }
        hi - lo
    };

    let before = z_span(&meshes, &handles);
    assert!(before > 0.0, "the stack has no depth to begin with");

    world.insert_resource(Sim {
        layers,
        ports: Vec::new(),
        gas: GasPlane::new(64, 64),
    });
    world.insert_resource(LayerCanvases { canvases, schematic });
    world.insert_resource(Perspective { depth_scale: 0.25, ..perspective });
    world.insert_resource(SlabGeometry {
        materials: Vec::new(),
        buf: MeshBuf::default(),
        built_depth: built,
    });
    world.insert_resource(meshes);

    let mut system = bevy::ecs::system::IntoSystem::into_system(rebuild_slab_geometry);
    system.initialize(&mut world);
    let _ = system.run((), &mut world);

    let meshes = world.resource::<Assets<Mesh>>();
    let after = z_span(meshes, &handles);
    let want = before * 0.25;
    assert!(
        (after - want).abs() < 0.5,
        "depth 0.25x should span {want:.1} px, geometry spans {after:.1} px (was {before:.1})"
    );
    assert_eq!(
        world.resource::<SlabGeometry>().built_depth,
        world.resource::<Perspective>().depth(),
        "the rebuild did not record the depth it built at, so it will rebuild every frame"
    );
}
