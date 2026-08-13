use common::prelude::*;

/// One chunk of one slab, as a piece of real geometry.
///
/// Geometry is per chunk rather than per slab so that a dirty chunk rebuilds only its own
/// mesh — which is what makes a 60 Hz per-pixel Plant affordable as 3D at all. Static slabs
/// build once and are never visited again.
#[derive(Component)]
pub struct SlabChunk {
    pub layer: usize,
    pub chunk: usize,
}

/// Marks the material owned by a slab, so the per-frame treatment update can find it.
#[derive(Component)]
pub struct SlabMaterial {
    pub layer: usize,
}

#[derive(Component)]
pub struct SchematicSprite;

#[derive(Component)]
pub struct ViewCamera;

/// The single directional light. LBP's depth reads as much from slabs shadowing each other
/// as from their spacing, so the rig is part of the renderer rather than the game's problem.
#[derive(Component)]
pub struct ViewSun;
