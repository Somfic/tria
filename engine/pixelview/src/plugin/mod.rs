use common::prelude::*;

mod components;
pub use components::*;

mod resources;
pub use resources::*;

mod systems;
pub use systems::*;

use crate::treat::RenderConfig as Cfg;

pub struct PixelViewPlugin {
    pub config: Cfg,
}

impl Default for PixelViewPlugin {
    fn default() -> Self {
        Self {
            config: Cfg::default(),
        }
    }
}

impl Plugin for PixelViewPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(self.config.clone())
            .init_resource::<Zoom>()
            .init_resource::<PixelCursor>()
            .init_resource::<ReachOrigin>()
            .init_resource::<LayerTransition>()
            .init_resource::<Perspective>()
            .init_resource::<ViewGeometry>()
            .insert_resource(XRay(false))
            // setup_view reads `Sim`, which pixelsim inserts in `build_layers`. Without
            // this the two Startup systems race and the view can be built against a
            // resource that does not exist yet.
            .add_systems(Startup, setup_view.after(pixelsim::build_layers))
            // Chained, not parallel: `sync_layer_sprites` reads the zoom and the
            // transition that the systems before it write, so leaving the order to the
            // scheduler would put the sprites a frame behind the camera at random.
            .add_systems(
                Update,
                (
                    apply_zoom,
                    tick_transition,
                    // geometry first: the camera, the cursor and the materials must all
                    // agree about where the slabs are *this* frame, not one behind
                    update_view_geometry,
                    update_camera,
                    update_cursor,
                    sync_slab_materials,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                (
                    update_view_geometry,
                    bake_layers,
                    upload_layers,
                    // geometry after the bake, because the bake is what records which
                    // chunks changed
                    rebuild_slab_geometry,
                )
                    .chain()
                    .after(pixelsim::step_sim),
            );
    }
}
