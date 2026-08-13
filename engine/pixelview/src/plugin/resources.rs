use common::prelude::*;

use bevy::pbr::StandardMaterial;

use crate::bake::LayerCanvas;
use crate::palette::Palette;

pub use crate::cursor::{PixelCursor, ReachOrigin};
pub use crate::depth::Perspective;
pub use crate::treat::RenderConfig;
pub use crate::zoom::Zoom;

#[derive(Resource)]
pub struct PaletteRes(pub Palette);

#[derive(Resource)]
pub struct LayerCanvases {
    pub canvases: Vec<LayerCanvas>,
    pub schematic: LayerCanvas,
}

#[derive(Resource, Debug, Clone)]
pub struct LayerTransition {
    pub from: usize,
    pub to: usize,
    pub t: f32,
}

impl Default for LayerTransition {
    fn default() -> Self {
        Self {
            from: 0,
            to: 0,
            t: 1.0,
        }
    }
}

#[derive(Resource)]
pub struct XRay(pub bool);

/// The slabs' materials, plus the scratch buffer their geometry is rebuilt through.
///
/// The buffer lives here rather than in the rebuild system so a churning Plant does not
/// allocate a vertex buffer per chunk per tick.
#[derive(Resource)]
pub struct SlabGeometry {
    pub materials: Vec<Handle<StandardMaterial>>,
    pub buf: crate::solid::MeshBuf,
    /// The z-step the meshes on screen were actually built at.
    ///
    /// Slab depth is a live knob, and a change to it moves every vertex of every slab — not
    /// only the chunks the sim dirtied. Without this the stack keeps the depth it was built
    /// with while the camera and the cursor move to the new one, which reads as the geometry
    /// detaching from the view.
    pub built_depth: f32,
}

/// Where the camera is and what that does to each plane, recomputed once a frame.
///
/// Everything that needs perspective reads this rather than deriving its own: the sprite
/// sync (scale and offset), the bake (how much low-pass a resampled plane needs) and the
/// cursor (where a screen position lands on a plane). One writer, three readers, so they
/// cannot disagree about where a plane is — a disagreement that shows up as a cursor
/// missing the pixels it is pointing at.
#[derive(Resource, Debug, Clone, Default)]
pub struct ViewGeometry {
    /// focal length in px, for the current viewport and fov
    pub focal_px: f32,
    /// camera-to-active-plane distance, in sim px
    pub distance: f32,
    /// on-screen scale of each layer relative to the active plane
    pub ratios: Vec<f32>,
    /// `z` of each layer, in sim px
    pub plane_z: Vec<f32>,
}

impl ViewGeometry {
    /// Scale of layer `i`. No geometry yet means a flat stack — which is what the
    /// pre-perspective renderer did, and what `Perspective::enabled == false` restores.
    #[inline]
    pub fn ratio(&self, i: usize) -> f32 {
        self.ratios.get(i).copied().unwrap_or(1.0)
    }

    /// Depth between two layers in sim px, positive when `behind` really is behind `front`.
    #[inline]
    pub fn gap(&self, front: usize, behind: usize) -> f32 {
        match (self.plane_z.get(front), self.plane_z.get(behind)) {
            (Some(a), Some(b)) => a - b,
            _ => 0.0,
        }
    }
}
