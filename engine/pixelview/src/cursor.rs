//! Screen -> cell conversion lives here and nowhere else. The game reads
//! `PixelCursor`; it never does the maths itself.
//!
//! # The coordinate convention, stated once
//!
//! * Cells are `(u16 x, u16 y)` with **`y = 0` at the top** and gravity toward `+y`,
//!   matching texture memory order and the sim's row order.
//! * A layer sprite is centred on the vessel origin with `custom_size = (w, h)` in world
//!   units, so **1 world unit == 1 sim pixel** and world `+y` is up. The flip between the
//!   two lives in exactly two functions below and nowhere else.
//! * Slabs are real geometry at real depths under a perspective camera, so a screen position
//!   is a **ray**, and the cell it means is where that ray meets the active slab's front
//!   face. [`plane_hit`] is that intersection; `update_cursor` applies it for the game, which
//!   is why the game reads `PixelCursor` and never converts anything itself.
//!
//! The version before this one called `viewport_to_world_2d` and then undid an orthographic
//! plane transform — correct for the flat renderer it was written for, and quietly wrong the
//! moment the camera became a `Camera3d`. It survived because nothing in the spikes aims at a
//! *particular* pixel: a sculpt brush landing somewhere near the mouse looks like a brush
//! working. Painting single materials in the sandbox is what made it obvious.

use bevy::prelude::{Resource, Vec2, Vec3};

#[derive(Resource, Debug, Clone, Default)]
pub struct PixelCursor {
    pub world: Vec2,
    /// in the ACTIVE layer, perspective-corrected
    pub cell: Option<(u16, u16)>,
    pub layer: usize,
    /// within `radius_px` of `ReachOrigin`, or reach disabled
    pub in_reach: bool,
}

#[derive(Resource, Debug, Clone)]
pub struct ReachOrigin {
    pub cell: (u16, u16),
    pub radius_px: f32,
    pub enabled: bool,
}

impl Default for ReachOrigin {
    fn default() -> Self {
        Self {
            cell: (0, 0),
            radius_px: 48.0,
            enabled: true,
        }
    }
}

/// Layer-local world position -> cell. `None` when outside the layer.
///
/// The vessel sits at the origin, so this is the `VesselFrame::origin == [0, 0]` case
/// written out. It is kept here rather than delegated to `pixelsim::VesselFrame` because
/// screen<->cell is this crate's contract, and the two must not be able to disagree about
/// where the centre is.
#[inline]
pub fn world_to_cell(world: Vec2, w: u16, h: u16) -> Option<(u16, u16)> {
    let px = world.x + w as f32 * 0.5;
    // world +y is up, cell +y is down
    let py = h as f32 * 0.5 - world.y;
    if px < 0.0 || py < 0.0 || px >= w as f32 || py >= h as f32 {
        return None;
    }
    Some((px as u16, py as u16))
}

/// Where a camera ray crosses the plane `z == plane_z`, in world xy.
///
/// `None` when the ray runs parallel to the plane, or when the plane is *behind* the camera —
/// both of which have to mean "no cell" rather than a number, because the arithmetic happily
/// produces a plausible-looking point for a plane the player cannot see.
#[inline]
pub fn plane_hit(origin: Vec3, dir: Vec3, plane_z: f32) -> Option<Vec2> {
    // the slab faces are all z = const, so the normal is Z and this is one divide
    if dir.z.abs() < 1.0e-6 {
        return None;
    }
    let t = (plane_z - origin.z) / dir.z;
    if t < 0.0 {
        return None;
    }
    Some((origin + dir * t).truncate())
}

/// Cell centre -> layer-local world position. The inverse of [`world_to_cell`].
#[inline]
pub fn cell_to_world(cell: (u16, u16), w: u16, h: u16) -> Vec2 {
    Vec2::new(
        cell.0 as f32 + 0.5 - w as f32 * 0.5,
        h as f32 * 0.5 - (cell.1 as f32 + 0.5),
    )
}

/// Chebyshev-free straight-line cell distance, in sim pixels.
#[inline]
pub fn cell_distance(a: (u16, u16), b: (u16, u16)) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;
    (dx * dx + dy * dy).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u16 = 1024;
    const H: u16 = 512;

    /// The one convention that, if wrong, silently mirrors the whole game vertically:
    /// cell y = 0 is the TOP row, and world +y is up.
    #[test]
    fn cell_y_zero_is_the_top_row() {
        let top = cell_to_world((0, 0), W, H);
        let bottom = cell_to_world((0, H - 1), W, H);
        assert!(top.y > bottom.y, "cell y=0 must be higher in world space");
        assert_eq!(top.y, H as f32 / 2.0 - 0.5);
    }

    #[test]
    fn world_and_cell_round_trip() {
        for cell in [(0u16, 0u16), (1, 1), (511, 255), (W - 1, H - 1), (7, 300)] {
            let world = cell_to_world(cell, W, H);
            assert_eq!(world_to_cell(world, W, H), Some(cell), "for {cell:?}");
        }
    }

    #[test]
    fn outside_the_vessel_is_none() {
        let half_w = W as f32 / 2.0;
        let half_h = H as f32 / 2.0;
        assert_eq!(world_to_cell(Vec2::new(-half_w - 1.0, 0.0), W, H), None);
        assert_eq!(world_to_cell(Vec2::new(half_w, 0.0), W, H), None);
        assert_eq!(world_to_cell(Vec2::new(0.0, half_h + 1.0), W, H), None);
        assert_eq!(world_to_cell(Vec2::new(0.0, -half_h - 1.0), W, H), None);
        // and the corners are inside
        assert!(world_to_cell(Vec2::new(-half_w, half_h - 0.5), W, H).is_some());
    }

    /// A perspective camera at a pan offset, aimed at a known cell: the ray back through
    /// that cell's screen position has to land on that cell again.
    ///
    /// This is the round trip the sandbox's brush needs and the one the old orthographic
    /// cursor failed. It is built from the projection directly rather than from anything in
    /// this crate, so it cannot agree with a bug by sharing one.
    #[test]
    fn a_camera_ray_lands_on_the_cell_it_was_aimed_at() {
        let f = crate::focal_px(900.0, 38.0);
        let plane_z = -48.0; // the front face of some slab back in the stack
        for cam_xy in [Vec2::ZERO, Vec2::new(137.0, -62.0)] {
            let cam = Vec3::new(cam_xy.x, cam_xy.y, plane_z + 260.0);
            for cell in [(0u16, 0u16), (5, 9), (511, 255), (W - 1, H - 1)] {
                let target = cell_to_world(cell, W, H);
                // where that point lands on screen, from the projection itself
                let screen = (target - cam.truncate()) * (f / (cam.z - plane_z));
                // and the ray back out through it: camera looks down -Z, so the ray's own
                // z component is -f in the same units the screen offset is measured in
                let dir = Vec3::new(screen.x, screen.y, -f);

                let hit = plane_hit(cam, dir, plane_z).expect("ray must reach the plane");
                assert!(
                    (hit - target).length() < 0.5,
                    "cell {cell:?} from camera {cam:?}: hit {hit:?} wanted {target:?}"
                );
                assert_eq!(world_to_cell(hit, W, H), Some(cell));
            }
        }
    }

    /// The depth of the plane is what makes the hit correct, so aiming the same ray at a
    /// deeper slab must land somewhere else. If this ever passes with the two equal, the
    /// cursor has stopped accounting for depth and is back to being flat.
    #[test]
    fn the_same_ray_hits_different_cells_on_different_slabs() {
        let cam = Vec3::new(0.0, 0.0, 300.0);
        let dir = Vec3::new(0.3, 0.2, -1.0);
        let near = plane_hit(cam, dir, 0.0).unwrap();
        let far = plane_hit(cam, dir, -24.0).unwrap();
        assert!(
            (far - near).length() > 1.0,
            "the ray hit both slabs in the same place: {near:?} vs {far:?}"
        );
        // and further along the ray is further from the axis it was fired down
        assert!(far.length() > near.length());
    }

    /// A plane behind the camera, or one the ray runs alongside, has to be "no cell". The
    /// arithmetic produces a perfectly plausible point for both, which is how a cursor ends
    /// up painting somewhere the player is not looking.
    #[test]
    fn unreachable_planes_have_no_cell() {
        let cam = Vec3::new(0.0, 0.0, 100.0);
        assert_eq!(plane_hit(cam, Vec3::new(0.0, 0.0, -1.0), 200.0), None);
        assert_eq!(plane_hit(cam, Vec3::new(1.0, 0.0, 0.0), 0.0), None);
    }
}
