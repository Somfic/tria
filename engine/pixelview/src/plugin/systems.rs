use bevy::asset::Assets;
use bevy::camera::{PerspectiveProjection, Projection};
use bevy::image::Image;
use bevy::light::{AmbientLight, DirectionalLight};
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::window::PrimaryWindow;
use common::prelude::*;
use pixelsim::{ActiveLayer, Materials, Sim};

use crate::bake::{BakeCtx, Cast, LayerCanvas};
use crate::cursor::{cell_distance, plane_hit, world_to_cell};
use crate::depth::{Perspective, active_distance, focal_px, shimmer_safe_blur_px};
use crate::palette::Palette;
use crate::schematic::bake_schematic;
use crate::slab::cast_offset;
use crate::solid::{MeshBuf, build_front, build_walls, slab_z};
use crate::treat::LayerTreatment;
use crate::zoom::{pixel_alpha, schematic_alpha};

use super::components::{SchematicSprite, SlabChunk, SlabMaterial, ViewCamera, ViewSun};
use super::resources::{
    LayerCanvases, LayerTransition, PaletteRes, PixelCursor, ReachOrigin, RenderConfig,
    SlabGeometry, ViewGeometry, XRay, Zoom,
};

/// How far in front of the frontmost slab the schematic overlay sits.
const SCHEMATIC_Z: f32 = 64.0;

/// Direction the sun comes from: upper left, slightly in front, so slabs cast down-right
/// onto the slab behind them.
///
/// Same convention the baked cues used, and the same one every cutaway illustration uses.
/// It is in front of the stack rather than level with it, or the frontmost slab would light
/// only its own edge and never reach the ones behind.
pub const SUN_DIR: Vec3 = Vec3::new(-0.45, -0.72, -0.53);

#[allow(clippy::too_many_arguments)]
pub fn setup_view(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    sim: Res<Sim>,
    materials: Res<Materials>,
    config: Res<RenderConfig>,
    active: Res<ActiveLayer>,
    perspective: Res<Perspective>,
) {
    // The camera is a real perspective camera now, dollied by `update_camera` to whatever
    // distance makes the active slab the size the player asked for. Slabs are solid, one
    // z-step deep and packed against each other, so depth is geometry all the way down and
    // the renderer no longer fakes any part of it.
    commands.spawn((
        Camera3d::default(),
        Projection::Perspective(PerspectiveProjection {
            fov: perspective.fov_deg.to_radians(),
            near: 1.0,
            far: 20_000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 0.0, 400.0),
        // Fill light, so a wall facing away from the sun is dark rather than black — the
        // vault's warmth cannot survive a scene lit by one hard light and nothing else. In
        // Bevy 0.18 this is a per-view component, not a resource.
        AmbientLight {
            color: Color::srgb(0.62, 0.68, 0.80),
            brightness: 900.0,
            ..default()
        },
        ViewCamera,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            shadows_enabled: true,
            // Shallow depth bias: our geometry is axis-aligned slabs with hard edges, where
            // the default bias is enough to detach a shadow from the wall casting it.
            shadow_depth_bias: 0.008,
            shadow_normal_bias: 0.6,
            ..default()
        },
        Transform::from_translation(-SUN_DIR.normalize() * 2_000.0)
            .looking_to(SUN_DIR.normalize(), Vec3::Y),
        ViewSun,
    ));

    let palette = Palette::from_table(&materials.0);
    let mut canvases = Vec::with_capacity(sim.layers.len());
    let mut slab_mats = Vec::with_capacity(sim.layers.len());

    for (i, layer) in sim.layers.iter().enumerate() {
        let depth = sim.depth_from(active.0, i);
        let treatment = config.treatment_for(i, depth, layer.slot, false);
        // Nearest sampling everywhere: the front faces are the pixel grid, and a linear
        // filter would smear it the moment the camera is anything but exactly 1:1.
        let canvas = LayerCanvas::new(&mut images, layer.w, layer.h, treatment.half_res, true);

        // One material per slab. `AlphaMode::Mask` is what makes air a hole rather than a
        // black pixel — including in the shadow pass, so a dug shaft lets light through.
        let mat = mats.add(StandardMaterial {
            base_color_texture: Some(canvas.handle.clone()),
            alpha_mode: AlphaMode::Mask(0.5),
            perceptual_roughness: 0.95,
            reflectance: 0.06,
            ..default()
        });

        let (front, back) = slab_z(i, perspective.depth());
        // One entity per chunk, so a dirty chunk rebuilds only its own geometry.
        for chunk in 0..layer.chunks.len() {
            let mut buf = MeshBuf::default();
            build_front(&mut buf, layer, chunk, front);
            build_walls(&mut buf, layer, chunk, front, back);
            commands.spawn((
                Mesh3d(meshes.add(buf.into_mesh())),
                MeshMaterial3d(mat.clone()),
                Transform::IDENTITY,
                SlabChunk { layer: i, chunk },
            ));
        }

        commands.spawn((SlabMaterial { layer: i }, MeshMaterial3d(mat.clone())));
        slab_mats.push(mat);
        canvases.push(canvas);
    }

    // The schematic stays a flat overlay — it is a diagram, not part of the scene — so it is
    // an unlit quad hung in front of the whole stack.
    let (sw, sh) = schematic_size(&sim);
    let schematic = LayerCanvas::new(&mut images, sw, sh, false, true);
    let (vw, vh) = (vessel_w(&sim) as f32, vessel_h(&sim) as f32);
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(vw, vh))),
        MeshMaterial3d(mats.add(StandardMaterial {
            base_color_texture: Some(schematic.handle.clone()),
            base_color: Color::srgba(1.0, 1.0, 1.0, 0.0),
            alpha_mode: AlphaMode::Blend,
            unlit: true,
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, SCHEMATIC_Z),
        SchematicSprite,
    ));

    commands.insert_resource(PaletteRes(palette));
    commands.insert_resource(SlabGeometry {
        materials: slab_mats,
        buf: MeshBuf::default(),
        built_depth: perspective.depth(),
    });
    commands.insert_resource(LayerCanvases {
        canvases,
        schematic,
    });
}

/// Rebuild the geometry of every chunk whose cells changed.
///
/// Runs after the bake, which is what records the dirty rects (the sim's dirty list has a
/// single reader and the bake is it). Static slabs report dirt exactly once, on their first
/// bake, so they mesh once and are never touched again — the property that makes real 3D with
/// shadows affordable at all, and the reason `One Simulated Plane` matters to the renderer as
/// much as to the solver.
pub fn rebuild_slab_geometry(
    mut geom: ResMut<SlabGeometry>,
    canvases: Res<LayerCanvases>,
    sim: Res<Sim>,
    perspective: Res<Perspective>,
    mut meshes: ResMut<Assets<Mesh>>,
    chunks: Query<(&SlabChunk, &Mesh3d)>,
) {
    if canvases.canvases.len() != sim.layers.len() {
        return;
    }
    // A depth change is not a dirty *region*, it is a dirty *stack*: every slab sits
    // somewhere else and every wall is a different length, so this one rebuilds the lot.
    // It happens when somebody turns the knob, so paying for a full remesh is fine.
    let depth = perspective.depth();
    let all = depth != geom.built_depth;

    // chunk indices that changed, per layer
    let mut dirty: Vec<Vec<usize>> = vec![Vec::new(); sim.layers.len()];
    for (i, canvas) in canvases.canvases.iter().enumerate() {
        let layer = &sim.layers[i];
        let cw = layer.chunks.cw as usize;
        for r in canvas.caster_rects() {
            let (cx0, cy0) = (r.0 / pixelsim::CHUNK_PX, r.1 / pixelsim::CHUNK_PX);
            let (cx1, cy1) = (r.2 / pixelsim::CHUNK_PX, r.3 / pixelsim::CHUNK_PX);
            for cy in cy0..=cy1 {
                for cx in cx0..=cx1 {
                    let c = cy as usize * cw + cx as usize;
                    if !dirty[i].contains(&c) {
                        dirty[i].push(c);
                    }
                }
            }
        }
    }
    if !all && dirty.iter().all(|d| d.is_empty()) {
        return;
    }

    geom.built_depth = depth;
    let SlabGeometry { buf, .. } = &mut *geom;
    for (slab, mesh) in &chunks {
        if !all && !dirty[slab.layer].contains(&slab.chunk) {
            continue;
        }
        let layer = &sim.layers[slab.layer];
        let (front, back) = slab_z(slab.layer, depth);
        buf.clear();
        build_front(buf, layer, slab.chunk, front);
        build_walls(buf, layer, slab.chunk, front, back);
        if let Some(handle) = meshes.get_mut(&mesh.0) {
            *handle = buf.into_mesh();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bake_layers(
    mut canvases: ResMut<LayerCanvases>,
    mut sim: ResMut<Sim>,
    materials: Res<Materials>,
    palette: Res<PaletteRes>,
    config: Res<RenderConfig>,
    active: Res<ActiveLayer>,
    transition: Res<LayerTransition>,
    xray: Res<XRay>,
    zoom: Res<Zoom>,
    time: Res<Time>,
    geom: Res<ViewGeometry>,
) {
    if canvases.canvases.len() != sim.layers.len() {
        return;
    }
    let dt = time.delta_secs();
    let (from, to, t) = visual_state(&active, &transition);

    // Front to back, because each layer casts onto the one behind it and a caster's mask
    // has to be current before the receiver bakes. Under x-ray the slab cues are off
    // entirely: lips and cast shadows are opaque depth cues, and x-ray exists precisely to
    // suspend depth ordering and show everything at once.
    let slabs = !xray.0;
    for i in 0..sim.layers.len() {
        let slot = sim.layers[i].slot;
        let treatment = treatment_of(&config, &xray, from, to, t, i, slot);

        // `half_res` lives on the canvas because it is the bake's *stride*, so it has to be
        // pushed across. Nothing else does: a change of stride, an x-ray toggle, or any
        // frame of the 0.3 s switch repaints every pixel and so cannot be expressed as
        // dirty tiles — and the bake detects all of that itself by fingerprinting the
        // treatment. Leaving it to the caller is how the x-ray toggle got missed.
        canvases.canvases[i].accum += dt;
        canvases.canvases[i].half_res = treatment.half_res;

        if treatment.bake_hz <= 0.0 || canvases.canvases[i].accum < 1.0 / treatment.bake_hz {
            continue;
        }

        // Three disjoint borrows of the canvas list: the slabs in front (read, for the
        // caster's mask), this one (written), and the one behind (invalidated where this
        // layer's shadow moved). They are different layers; `split_at_mut` is how the
        // borrow checker is shown that.
        let (front_canvases, from_here) = canvases.canvases.split_at_mut(i);
        let (here, behind) = from_here.split_at_mut(1);
        let canvas = &mut here[0];
        let caster = if slabs && config.shadow.strength > 0.0 {
            front_canvases.last().map(|c| Cast {
                mask: &c.mask,
                cfg: config.shadow,
                // one slab of gap, uniformly — the stack is contiguous now. This whole
                // path is off by default; real geometry casts real shadows.
                offset: cast_offset(&config.shadow, config.extrude.dir, 1.0),
            })
        } else {
            None
        };

        let lip = if slabs && config.extrude.px > 0.0 {
            // a cavity is a seam, not a room: half the lip, so it does not read as a slab
            let scale = if matches!(
                slot,
                pixelsim::LayerSlot::Cavity | pixelsim::LayerSlot::Cavity
            ) {
                0.5
            } else {
                1.0
            };
            Some((config.extrude, config.extrude.px * scale))
        } else {
            None
        };

        canvas.bake_ctx(
            &mut sim.layers[i],
            &BakeCtx {
                table: &materials.0,
                pal: &palette.0,
                treat: &treatment,
                tint_sleeping: config.tint_sleeping,
                outline: xray.0,
                extrude: lip,
                cast: caster,
                // a *minified* plane gets its low-pass whether or not the art asked for
                // one; zoomed in, where every plane is magnified, this is 0
                min_blur_px: shimmer_safe_blur_px(
                    crate::zoom::snapped(zoom.current) * geom.ratio(i),
                ),
            },
        );

        // This layer is now a caster for the next one: refresh its occupancy, and tell the
        // layer behind that its shadow moved wherever this layer's cells did. Without that
        // second half the cast would go stale in exactly the tiles it should be tracking.
        if slabs && config.shadow.strength > 0.0 {
            canvas.update_mask(&sim.layers[i], config.shadow.softness);
            if let Some(next) = behind.first_mut() {
                for r in canvas.caster_rects() {
                    next.mark_dirty_rect(*r);
                }
            }
        }
    }

    // the schematic seam: only paid for once the crossfade has actually begun
    if schematic_alpha(zoom.current) > 0.0 {
        canvases.schematic.clear();
        // back to front, because `bake_schematic` composites source-over
        for i in (0..sim.layers.len()).rev() {
            let depth = sim.depth_from(to, i);
            bake_schematic(&mut canvases.schematic, &sim.layers[i], &materials.0, depth);
        }
    }
}

pub fn upload_layers(mut canvases: ResMut<LayerCanvases>, mut images: ResMut<Assets<Image>>) {
    // Bevy 0.18 re-uploads a modified image whole, so only canvases that actually baked
    // this tick are pushed — that is what makes the 15 Hz rear-layer bake rate pay off
    // twice. `upload` then copies only the rows that bake touched, so a settled vessel with
    // one changed tile does not memcpy 2 MB per layer either. `ResMut` rather than `Res`
    // because tracking which rows are owed is the canvas's own state.
    let LayerCanvases {
        canvases: layers,
        schematic,
    } = &mut *canvases;
    for canvas in layers.iter_mut() {
        if canvas.needs_upload() {
            canvas.upload(&mut images);
        }
    }
    if schematic.needs_upload() {
        schematic.upload(&mut images);
    }
}

/// Per-frame material state: how solid each slab is, and the schematic crossfade.
///
/// Geometry no longer moves — the slabs are fixed solids in world space and the *camera*
/// moves instead — so this only touches materials. A slab in front of the one the player is
/// working in has to thin out, or it walls them out of their own workspace; LBP can lean on
/// its camera for that and a fixed cutaway cannot.
pub fn sync_slab_materials(
    slabs: Query<(&SlabMaterial, &MeshMaterial3d<StandardMaterial>)>,
    schematic: Query<&MeshMaterial3d<StandardMaterial>, With<SchematicSprite>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    config: Res<RenderConfig>,
    active: Res<ActiveLayer>,
    transition: Res<LayerTransition>,
    xray: Res<XRay>,
    zoom: Res<Zoom>,
    sim: Res<Sim>,
) {
    let (from, to, t) = visual_state(&active, &transition);
    let pa = pixel_alpha(zoom.current);

    for (slab, handle) in &slabs {
        let Some(layer) = sim.layers.get(slab.layer) else {
            continue;
        };
        let treatment = treatment_of(&config, &xray, from, to, t, slab.layer, layer.slot);
        let Some(mat) = mats.get_mut(&handle.0) else {
            continue;
        };
        let alpha = (treatment.alpha * pa).clamp(0.0, 1.0);
        mat.base_color = Color::srgba(1.0, 1.0, 1.0, alpha);
        // Mask while opaque, Blend once it is a scrim. Blending everything would cost sorting
        // and lose the shadow silhouette; masking a translucent slab would ignore the alpha
        // entirely and keep walling the player out.
        mat.alpha_mode = if alpha >= 0.999 {
            AlphaMode::Mask(0.5)
        } else {
            AlphaMode::Blend
        };
    }

    if let Some(handle) = schematic.iter().next() {
        if let Some(mat) = mats.get_mut(&handle.0) {
            mat.base_color = Color::srgba(1.0, 1.0, 1.0, schematic_alpha(zoom.current));
        }
    }
}

/// Dolly the camera so the active slab is exactly `zoom` screen px per cell.
///
/// This is the whole of the zoom now: a perspective camera at the distance that makes the
/// plane being sculpted pixel-exact. Everything else follows from geometry — slabs behind
/// come out smaller and parallaxed because they *are* further away, not because anything
/// scales them.
pub fn update_camera(
    mut camera: Query<(&mut Transform, &mut Projection), With<ViewCamera>>,
    geom: Res<ViewGeometry>,
    perspective: Res<Perspective>,
    active: Res<ActiveLayer>,
) {
    let (front, _) = slab_z(active.0, perspective.depth());
    for (mut transform, mut projection) in &mut camera {
        transform.translation.z = front + geom.distance;
        if let Projection::Perspective(p) = &mut *projection {
            p.fov = perspective.fov_deg.to_radians();
        }
    }
}

/// Recompute [`ViewGeometry`] for this frame: focal length from the viewport, distance
/// from the zoom, then a scale per plane.
///
/// This is the camera dolly. The camera sits wherever it must for the *active* plane to be
/// exactly `zoom` screen px per cell — so the plane being sculpted is always pixel-exact —
/// and every other plane's scale falls out of its distance. Zoom in and the stack opens up;
/// pull out to vessel scale and it flattens toward the diagram, which is the telephoto
/// behaviour `Scale and Zoom` wants at that end.
pub fn update_view_geometry(
    mut geom: ResMut<ViewGeometry>,
    perspective: Res<Perspective>,
    zoom: Res<Zoom>,
    sim: Res<Sim>,
    active: Res<ActiveLayer>,
    window: Query<&Window, With<PrimaryWindow>>,
) {
    let viewport_h = window
        .single()
        .ok()
        .map(|w| w.resolution.height())
        .unwrap_or(1080.0);

    let f = focal_px(viewport_h, perspective.fov_deg);
    let d = active_distance(f, crate::zoom::snapped(zoom.current));
    geom.focal_px = f;
    geom.distance = d;

    // Slab front faces, from the contiguous stack: slab i's front is -(i * depth), and its
    // back is the next slab's front. The geometry builder uses the same function, so the
    // cursor, the camera and the meshes cannot disagree about where a slab is.
    geom.plane_z.clear();
    for i in 0..sim.layers.len() {
        geom.plane_z.push(slab_z(i, perspective.depth()).0);
    }

    // Kept for the HUD and the spikes' markers: what each slab's apparent scale works out
    // to. Nothing in the renderer applies it any more — the camera does the projecting.
    let active_z = geom.plane_z.get(active.0).copied().unwrap_or(0.0);
    geom.ratios.clear();
    for i in 0..sim.layers.len() {
        let gap = active_z - geom.plane_z[i];
        geom.ratios.push(if gap.abs() < 1.0e-6 {
            1.0
        } else {
            d / (d + gap)
        });
    }
}

pub fn apply_zoom(mut zoom: ResMut<Zoom>, time: Res<Time>) {
    let dt = time.delta_secs();
    let (min, max, smoothing) = (zoom.min, zoom.max, zoom.smoothing);
    zoom.target = zoom.target.clamp(min, max);

    // frame-rate independent exponential approach
    let k = if smoothing > 0.0 {
        1.0 - (-smoothing * dt).exp()
    } else {
        1.0
    };
    let current = zoom.current + (zoom.target - zoom.current) * k;
    zoom.current = current.clamp(min, max);

    // The projection is untouched: zoom is a camera *distance* now, applied by
    // `update_camera` from the geometry this frame. A perspective camera that changed its
    // fov to zoom would warp the cutaway as it went.
}

pub fn update_cursor(
    mut cursor: ResMut<PixelCursor>,
    reach: Res<ReachOrigin>,
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<ViewCamera>>,
    sim: Res<Sim>,
    active: Res<ActiveLayer>,
    perspective: Res<Perspective>,
) {
    cursor.layer = active.0;

    let Ok(window) = window.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };
    let Some(viewport) = window.cursor_position() else {
        cursor.cell = None;
        cursor.in_reach = false;
        return;
    };
    // A screen position under a perspective camera is a ray, not a point. The cell it means
    // is where that ray meets the face the player is looking at: the FRONT of the active
    // slab, which is the surface the pixels are drawn on.
    let Ok(ray) = camera.viewport_to_world(camera_transform, viewport) else {
        cursor.cell = None;
        cursor.in_reach = false;
        return;
    };
    let Some(layer) = sim.layers.get(active.0) else {
        cursor.cell = None;
        cursor.in_reach = false;
        return;
    };
    let plane_z = slab_z(active.0, perspective.depth()).0;
    let Some(world) = plane_hit(ray.origin, *ray.direction, plane_z) else {
        cursor.cell = None;
        cursor.in_reach = false;
        return;
    };
    cursor.world = world;
    cursor.cell = world_to_cell(world, layer.w, layer.h);

    cursor.in_reach = match cursor.cell {
        Some(cell) => !reach.enabled || cell_distance(cell, reach.cell) <= reach.radius_px,
        None => false,
    };
}

pub fn tick_transition(
    mut transition: ResMut<LayerTransition>,
    config: Res<RenderConfig>,
    time: Res<Time>,
) {
    if transition.t >= 1.0 {
        return;
    }
    if config.transition_secs <= 0.0 {
        transition.t = 1.0;
    } else {
        transition.t = (transition.t + time.delta_secs() / config.transition_secs).min(1.0);
    }
    if transition.t >= 1.0 {
        transition.from = transition.to;
    }
}

/// Game-callable helper: start the 0.3 s cross-animation to a new active layer.
///
/// `ActiveLayer` updates immediately — gameplay and sculpting act on the new layer at once
/// — while the *visual* switch is what animates. The 0.3 s exposure the vault asks for is
/// a gameplay commitment, not a rendering delay.
pub fn request_active_layer(active: &mut ActiveLayer, transition: &mut LayerTransition, to: usize) {
    if active.0 == to && transition.t >= 1.0 {
        return;
    }
    transition.from = active.0;
    transition.to = to;
    transition.t = 0.0;
    active.0 = to;
}

/// `(from, to, t)` for the visual cross-animation. Falls back to `ActiveLayer` once the
/// animation is done, so a game that sets `ActiveLayer` directly still renders correctly.
#[inline]
fn visual_state(active: &ActiveLayer, transition: &LayerTransition) -> (usize, usize, f32) {
    if transition.t < 1.0 {
        (transition.from, transition.to, transition.t.clamp(0.0, 1.0))
    } else {
        (active.0, active.0, 1.0)
    }
}

/// The treatment for layer `i`, cross-animated between the outgoing and incoming active
/// layer. Both endpoints come from `treatment_for`, so the animation cannot drift away
/// from the static rule.
#[inline]
fn treatment_of(
    config: &RenderConfig,
    xray: &XRay,
    from: usize,
    to: usize,
    t: f32,
    i: usize,
    slot: pixelsim::LayerSlot,
) -> LayerTreatment {
    let depth_to = i as i32 - to as i32;
    let target = config.treatment_for(i, depth_to, slot, xray.0);
    if t >= 1.0 {
        return target;
    }
    let depth_from = i as i32 - from as i32;
    let source = config.treatment_for(i, depth_from, slot, xray.0);
    LayerTreatment::lerp(source, target, t)
}

#[inline]
fn vessel_w(sim: &Sim) -> u16 {
    sim.layers.first().map(|l| l.w).unwrap_or(1)
}

#[inline]
fn vessel_h(sim: &Sim) -> u16 {
    sim.layers.first().map(|l| l.h).unwrap_or(1)
}

/// one coarse texel per `CELL_PX` block, rounded up
#[inline]
fn schematic_size(sim: &Sim) -> (u16, u16) {
    let w = vessel_w(sim).div_ceil(pixelsim::CELL_PX).max(1);
    let h = vessel_h(sim).div_ceil(pixelsim::CELL_PX).max(1);
    (w, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::IntoSystem;

    /// Initialising a system is where Bevy validates parameter access and panics on
    /// conflicting queries — e.g. two `&mut Sprite` queries whose filters cannot be proven
    /// disjoint. That is a startup panic rather than a compile error, and the sim's solver
    /// bodies are not implemented yet so the app cannot be booted to find it. Initialising
    /// each system against an empty world catches the whole class without running them
    /// (resources are only required at *run* time, not at init).
    fn assert_initialises<M, S: IntoSystem<(), (), M>>(system: S) {
        let mut world = World::new();
        let mut system = IntoSystem::into_system(system);
        system.initialize(&mut world);
    }

    #[test]
    fn every_system_has_compatible_parameter_access() {
        assert_initialises(setup_view);
        assert_initialises(bake_layers);
        assert_initialises(upload_layers);
        assert_initialises(rebuild_slab_geometry);
        assert_initialises(sync_slab_materials);
        assert_initialises(update_camera);
        assert_initialises(apply_zoom);
        assert_initialises(update_cursor);
        assert_initialises(tick_transition);
    }

    /// Draw order used to be a hand-managed sprite `z` with a bias so the active layer won
    /// ties. There is no such thing now: the slabs are solids at real depths and the depth
    /// buffer sorts them, which is one whole class of bug — "which sprite is in front" —
    /// deleted rather than fixed. What has to hold instead is that the stack recedes and
    /// stays contiguous, which `solid::slabs_abut_with_no_gap` covers.
    #[test]
    fn the_stack_recedes_from_the_camera() {
        let d = crate::solid::SLAB_DEPTH_PX;
        // index 0 is the frontmost slab, so front-face z decreases with index
        assert!(slab_z(0, d).0 > slab_z(1, d).0);
        assert!(slab_z(1, d).0 > slab_z(2, d).0);
        // and every slab is behind its own front face
        for i in 0..4 {
            let (front, back) = slab_z(i, d);
            assert!(back < front);
        }
    }

    #[test]
    fn visual_state_falls_back_to_active_layer_when_settled() {
        let active = ActiveLayer(2);
        let settled = LayerTransition {
            from: 0,
            to: 0,
            t: 1.0,
        };
        assert_eq!(visual_state(&active, &settled), (2, 2, 1.0));

        let animating = LayerTransition {
            from: 1,
            to: 2,
            t: 0.5,
        };
        assert_eq!(visual_state(&active, &animating), (1, 2, 0.5));
    }
}
