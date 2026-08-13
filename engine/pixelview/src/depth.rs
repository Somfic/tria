//! Where the layers actually sit in Z, and what a perspective camera does to them.
//!
//! # The design change this module encodes
//!
//! The vault used to say "the camera never moves in Z; your layer change reads through
//! depth-of-field, lighting and contrast". That made depth a *focus state*: the layer you
//! stood in was bright and sharp, the rest were dimmed, and the stack read as separate
//! images you switched between rather than one space seen at once.
//!
//! LittleBigPlanet does the opposite. Its layers are slabs at real depths under a
//! perspective camera that dollies, so a rear plane is genuinely smaller and genuinely
//! parallaxed, and it looks the same whether or not you are standing in it. Depth is
//! geometry; the active layer is a light touch on top. See `Layers and Cavities`.
//!
//! # Perspective without a 3D camera, exactly
//!
//! Every layer is a flat quad parallel to the image plane. For such a plane the
//! perspective projection collapses to a **uniform scale plus a camera-relative
//! translation** — no approximation, no residual. With focal length `f` (px), camera at
//! `(cx, cy, cz)` looking down `-Z`, a point `(px, py)` on the plane at `z` lands at
//!
//! ```text
//! screen_x = (px - cx) * f / (cz - z)
//! ```
//!
//! so a plane's on-screen scale is `f / (cz - z)` and nothing else. Writing `d` for the
//! active plane's distance `cz - z_active` and `gap` for how far a plane sits *behind*
//! the active one, that scale relative to the active plane is
//!
//! ```text
//! ratio = d / (d + gap)
//! ```
//!
//! which is [`plane_ratio`]. An orthographic camera showing `zoom` screen px per world
//! unit reproduces the perspective image pixel-for-pixel by drawing each plane at
//! `scale = ratio` and `translation = camera_xy * (1 - ratio)` — see [`plane_transform`],
//! and `matches_a_true_perspective_projection` for the proof against a hand-rolled 3D
//! projection.
//!
//! Two things fall out of doing it this way rather than with a `Camera3d`:
//!
//! * The active plane stays **pixel-exact** at `ratio == 1.0`, so sculpting keeps its
//!   integer cell grid and `Spike 6` is unaffected. Under a real 3D camera every plane
//!   including the active one lands on a fractional texel scale.
//! * Rear planes are resampled — that is what perspective *is* — and are the only things
//!   that can shimmer, and then only when the view *minifies* them below one screen px per
//!   cell. [`shimmer_safe_blur_px`] states the low-pass that case needs; zoomed in, where
//!   every plane is magnified, it correctly asks for none.
//!
//! What this cannot do is show a plane at an *angle*, so slab side faces are baked into
//! the layer texture rather than being real geometry (`bake::extrude`). At our depths that
//! is what LBP's faces look like anyway: a few px of darker lip on one side.
//!
//! # The dolly
//!
//! `d` comes from the zoom the player asked for: `d = f / zoom`. Zoomed in, `d` is small,
//! the gaps are a large fraction of it, and perspective is strong. Zoomed out to vessel
//! scale, `d` is large and the stack flattens toward a diagram — which is exactly the
//! telephoto flattening a real camera would give, and exactly what `Scale and Zoom` wants
//! at the schematic end.

use bevy::prelude::{Resource, Vec2};

/// The old per-slot depth table lived here: thin slots 8px, thick slots 24px, with a `z`
/// derived from a slot's position in a mixed-thickness stack.
///
/// It is gone. Slabs are now **uniform and contiguous** — one z-step each, back face of one
/// against the front face of the next — which is what makes the stack read as a solid block
/// of layers rather than planes at assorted depths, and what "each is 1 z deep" means. The
/// single source of truth for where a slab sits is [`crate::solid::slab_z`]; nothing derives
/// depth from slot identity any more, so nothing can disagree about it.
///
/// Slot *thickness* as a gameplay property — who can stand where, who has to crouch — never
/// belonged here anyway and lives on `pixelsim::LayerSlot`.

#[derive(Resource, Debug, Clone, Copy)]
pub struct Perspective {
    /// Vertical field of view, degrees. Narrow: this is a cutaway, not a first-person
    /// view, and a wide lens bends the vessel's verticals in a way the anti-miniature
    /// look cannot afford.
    pub fov_deg: f32,
    /// Multiplies the z-step. 1.0 is the vault's literal stack; below 1.0 flattens it, above
    /// 1.0 exaggerates. The one knob to turn when the depth cue reads wrong, and the reason
    /// slab depth is a resource rather than only a constant.
    pub depth_scale: f32,
    /// Off collapses the stack almost flat — see [`Perspective::depth`]. Kept as the honest
    /// A/B for the legibility spike: if the depth work is doing nothing, this shows it.
    pub enabled: bool,
    /// One z-step, in sim px: both a slab's depth and the spacing between slabs, since the
    /// stack is contiguous. `[` and `]` in the spikes scale this.
    pub slab_depth_px: f32,
}

impl Default for Perspective {
    fn default() -> Self {
        Self {
            fov_deg: 38.0,
            depth_scale: 1.0,
            enabled: true,
            slab_depth_px: crate::solid::SLAB_DEPTH_PX,
        }
    }
}

/// Slab depth used when perspective is switched off.
///
/// Not zero: four coplanar slabs would z-fight, and a flickering stack is not a useful
/// control. Half a sim px is flat to the eye and still orders the slabs unambiguously.
pub const FLAT_DEPTH_PX: f32 = 0.5;

impl Perspective {
    /// The z-step actually used this frame. One place, so the camera, the geometry builder
    /// and the cursor cannot disagree about how deep a slab is.
    ///
    /// `depth_scale` multiplies here rather than at any call site. It used to be applied to a
    /// per-slot depth table that no longer exists, which left the knob wired to nothing: the
    /// spike's `F5`/`F6` changed the number, logged it, and moved not one vertex.
    #[inline]
    pub fn depth(&self) -> f32 {
        if self.enabled {
            (self.slab_depth_px * self.depth_scale).max(FLAT_DEPTH_PX)
        } else {
            FLAT_DEPTH_PX
        }
    }
}

/// Focal length in px for a viewport `viewport_h` px tall.
#[inline]
pub fn focal_px(viewport_h: f32, fov_deg: f32) -> f32 {
    let half = (fov_deg.to_radians() * 0.5).tan().max(1.0e-4);
    viewport_h * 0.5 / half
}

/// Distance from the camera to the active plane, in world units (== sim px), for a zoom
/// expressed as screen px per sim px.
///
/// This is the whole dolly: the camera sits wherever it must to make the active plane
/// exactly `zoom` px per cell, so the plane the player is sculpting is always pixel-exact
/// and every other plane's scale follows from geometry.
#[inline]
pub fn active_distance(focal_px: f32, zoom: f32) -> f32 {
    (focal_px / zoom.max(1.0e-4)).max(1.0)
}

/// On-screen scale of a plane `gap` sim px *behind* the active plane (negative for planes
/// in front of it), relative to the active plane.
///
/// Clamped at the near side so a plane in front of the camera cannot invert or blow up:
/// `MAX_FRONT_RATIO` is the widest a front plane may get, which at the shipped stack is
/// only ever reached if `depth_scale` is pushed far past 1.
#[inline]
pub fn plane_ratio(active_distance: f32, gap: f32) -> f32 {
    let d = active_distance.max(1.0);
    (d / (d + gap).max(d / MAX_FRONT_RATIO)).min(MAX_FRONT_RATIO)
}

/// A front plane may not be drawn more than this much larger than the active plane.
pub const MAX_FRONT_RATIO: f32 = 3.0;

/// The `(translation, scale)` an orthographic camera needs to draw a plane of the given
/// `ratio` exactly where a perspective camera would put it.
#[inline]
pub fn plane_transform(camera: Vec2, ratio: f32) -> (Vec2, f32) {
    (camera * (1.0 - ratio), ratio)
}

/// `plane_local` — the inverse of [`plane_transform`] — used to live here, and was how a
/// screen position became a cell.
///
/// It is gone, deliberately. Slabs are real geometry now, so a screen position is a ray and
/// the cell it means is where that ray meets a slab's front face
/// ([`crate::cursor::plane_hit`]). Keeping a tested-looking orthographic inverse around is
/// what let the cursor go on undoing a transform the renderer had stopped applying: the brush
/// landed near the mouse rather than under it, and "near" is invisible until you are placing
/// single pixels.

/// The blur radius a plane needs so that resampling it cannot shimmer.
///
/// The argument that matters is **screen px per cell** — `zoom * ratio` — not the ratio on
/// its own. Aliasing is a minification problem: it needs more than one texel landing on a
/// screen pixel. A plane being drawn at 5 screen px per cell is oversampled by 5x and
/// cannot alias no matter how far back it sits.
///
/// The first version of this took only `ratio` and blurred every plane that was not the
/// active one. At the shipped default zoom of 6 that put 2-3 px of *texture* blur on planes
/// that magnification then smeared across ~15 screen px, which turned the entire cutaway
/// into grey mush — and, being keyed to a value that does not change with zoom, did it at
/// every zoom level. Blur is measured in texture px; whether that is visible depends
/// entirely on the magnification it is then viewed through, so the trigger has to be the
/// screen scale.
#[inline]
pub fn shimmer_safe_blur_px(screen_px_per_cell: f32) -> f32 {
    // one texel per screen px is the Nyquist limit; above it there is nothing to prefilter
    if screen_px_per_cell >= 1.0 {
        return 0.0;
    }
    // one px of blur per texel crowded onto a screen pixel, floored at the radius below
    // which the bake ignores blur entirely (so "a little undersampled" still gets filtered)
    let undersample = 1.0 / screen_px_per_cell.max(1.0e-4);
    undersample.max(crate::bake::BLUR_MIN_PX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The claim the whole module rests on: drawing each plane at `plane_transform`'s
    /// scale and offset under an orthographic camera is not an approximation of
    /// perspective, it *is* perspective, for planes parallel to the image plane.
    #[test]
    fn matches_a_true_perspective_projection() {
        let (viewport_h, fov, zoom) = (900.0, 38.0, 4.0);
        let f = focal_px(viewport_h, fov);
        let d = active_distance(f, zoom);
        let cam = Vec2::new(137.0, -42.0);

        for gap in [-16.0, 0.0, 8.0, 24.0, 96.0] {
            let ratio = plane_ratio(d, gap);
            let (offset, scale) = plane_transform(cam, ratio);
            for p in [
                Vec2::new(0.0, 0.0),
                Vec2::new(512.0, 256.0),
                Vec2::new(-311.0, 77.0),
            ] {
                // what a real 3D camera would do: cz - z == d + gap
                let truth = (p - cam) * (f / (d + gap));
                // what we do: ortho at `zoom` px per unit, plane drawn scaled and offset
                let ours = ((offset + p * scale) - cam) * zoom;
                assert!(
                    (truth - ours).length() < 1.0e-3,
                    "gap {gap} point {p:?}: perspective {truth:?} vs ortho {ours:?}"
                );
            }
        }
    }

    /// Pixel-exactness on the plane being sculpted is the property that lets Spike 6
    /// keep working, so it is worth its own test.
    /// The active plane is drawn at exactly the scale the player asked for, at every zoom —
    /// that is what keeps sculpting on an integer grid. It needs no prefilter of its own
    /// while the view magnifies; pulled out past 1:1 the whole view is minified and it gets
    /// the same low-pass as everything else, because at that point *it* aliases too.
    #[test]
    fn the_active_plane_is_drawn_at_exactly_the_asked_for_scale() {
        let f = focal_px(1080.0, 38.0);
        for zoom in [0.5, 1.0, 2.0, 6.0, 8.0] {
            let d = active_distance(f, zoom);
            assert_eq!(plane_ratio(d, 0.0), 1.0, "at zoom {zoom}");
            let blur = shimmer_safe_blur_px(zoom * plane_ratio(d, 0.0));
            if zoom >= 1.0 {
                assert_eq!(blur, 0.0, "magnified active plane blurred at zoom {zoom}");
            } else {
                assert!(blur > 0.0, "minified active plane unfiltered at zoom {zoom}");
            }
        }
    }

    /// The regression that turned the whole cutaway to mush: a magnified plane cannot
    /// alias, so it must not be prefiltered however far back it sits.
    #[test]
    fn magnified_planes_are_never_blurred_for_shimmer() {
        let f = focal_px(1736.0, 38.0);
        for zoom in [2.0, 4.0, 6.0, 8.0] {
            let d = active_distance(f, zoom);
            for gap in [8.0, 24.0, 56.0, 96.0] {
                let scale = zoom * plane_ratio(d, gap);
                if scale >= 1.0 {
                    assert_eq!(
                        shimmer_safe_blur_px(scale),
                        0.0,
                        "zoom {zoom}, gap {gap}: {scale:.2} screen px per cell was blurred"
                    );
                }
            }
        }
    }

    #[test]
    fn rear_planes_shrink_front_planes_grow_and_order_is_strict() {
        let d = active_distance(focal_px(900.0, 38.0), 4.0);
        let behind_1 = plane_ratio(d, 8.0);
        let behind_2 = plane_ratio(d, 32.0);
        let front = plane_ratio(d, -8.0);
        assert!(behind_2 < behind_1, "further must be smaller");
        assert!(behind_1 < 1.0);
        assert!(front > 1.0, "a plane in front must be larger");
    }

    /// Zooming in must strengthen the depth cue and zooming out must flatten it, or the
    /// schematic end of `Scale and Zoom` fights the perspective.
    #[test]
    fn perspective_strengthens_as_the_camera_closes_in() {
        let f = focal_px(900.0, 38.0);
        // the whole stack: four slabs, one z-step each
        let gap = 4.0 * crate::solid::SLAB_DEPTH_PX;
        let close = plane_ratio(active_distance(f, 8.0), gap);
        let far = plane_ratio(active_distance(f, 0.4), gap);
        assert!(close < far, "close-up must foreshorten more: {close} vs {far}");
        // the diagram end of the zoom is flat enough that perspective cannot fight the
        // schematic for legibility
        assert!(far > 0.95, "vessel scale should be nearly flat, got {far}");
        assert!(close < 0.9, "close-up should be clearly foreshortened, got {close}");
    }

    /// A *minified* plane must always get at least the bake's minimum blur, or nearest
    /// sampling crawls as the camera pans — the one real cost of true perspective on a
    /// pixel grid, and it only applies pulled out past 1:1.
    #[test]
    fn minified_planes_get_a_low_pass() {
        for screen_px_per_cell in [0.99, 0.75, 0.5, 0.25] {
            assert!(shimmer_safe_blur_px(screen_px_per_cell) >= crate::bake::BLUR_MIN_PX);
        }
        // and more undersampling asks for more blur
        assert!(shimmer_safe_blur_px(0.25) > shimmer_safe_blur_px(0.9));
    }
}
