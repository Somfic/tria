//! Zoom is expressed as screen px per sim px. Applied by mutating `Projection`,
//! never `Transform`.
//!
//! The vault ([[Scale and Zoom]]) asks for "one continuous zoom that changes
//! representation as it goes" across three named scales, so the mapping is:
//!
//! | zoom (screen px / sim px) | vault scale | representation |
//! |---|---|---|
//! | 8 .. 2       | Local     | per-pixel, integer-snapped so pixels stay square |
//! | 2 .. 0.75    | Deck      | per-pixel, free scale — the Scarry cutaway |
//! | 0.75 .. 0.40 | crossfade | pixels dissolve into the schematic |
//! | 0.40 .. 0.20 | Vessel    | schematic only |
//!
//! Schematic rendering itself is out of scope for the slice; the crossfade weights here
//! plus `schematic.rs` are the seam it will land in.

use bevy::prelude::Resource;

#[derive(Resource, Debug, Clone)]
pub struct Zoom {
    pub target: f32,
    pub current: f32,
    pub min: f32,
    pub max: f32,
    pub smoothing: f32,
}

impl Default for Zoom {
    fn default() -> Self {
        Self {
            target: 6.0,
            current: 6.0,
            min: 0.20,
            max: 8.0,
            smoothing: 12.0,
        }
    }
}

/// below this zoom, sculpting is disabled
pub const SCULPT_MIN_ZOOM: f32 = 2.0;
/// above this zoom, the pixel view is fully opaque
pub const PIXEL_FULL_ZOOM: f32 = 0.75;
/// below this zoom, the pixel view is gone and only the schematic remains
pub const PIXEL_GONE_ZOOM: f32 = 0.40;

/// integer snap (6, 5, 4, 3, 2) above 2.0; free scale below
///
/// Snapping matters only while individual pixels are visible: a fractional scale there
/// gives pixels uneven screen footprints and the sculpting grid stops being trustworthy,
/// which is fatal for the "shave two pixels off a funnel wall" test. Below 2x the pixels
/// are too small to alias visibly, and free scale buys a smooth pull-out to vessel scale.
#[inline]
pub fn snapped(zoom: f32) -> f32 {
    if zoom >= SCULPT_MIN_ZOOM {
        zoom.round()
    } else {
        zoom
    }
}

/// 1.0 above 0.75, 0.0 below 0.40, linear between
#[inline]
pub fn pixel_alpha(zoom: f32) -> f32 {
    ((zoom - PIXEL_GONE_ZOOM) / (PIXEL_FULL_ZOOM - PIXEL_GONE_ZOOM)).clamp(0.0, 1.0)
}

/// the complement of `pixel_alpha`, for the schematic crossfade
#[inline]
pub fn schematic_alpha(zoom: f32) -> f32 {
    1.0 - pixel_alpha(zoom)
}

#[inline]
pub fn ortho_scale(zoom: f32) -> f32 {
    1.0 / snapped(zoom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapping_keeps_pixels_square_only_where_it_matters() {
        assert_eq!(snapped(6.2), 6.0);
        assert_eq!(snapped(4.6), 5.0);
        assert_eq!(snapped(2.0), 2.0);
        // below the sculpt threshold, free scale
        assert_eq!(snapped(1.4), 1.4);
        assert_eq!(snapped(0.31), 0.31);
    }

    #[test]
    fn ortho_scale_is_the_reciprocal_of_screen_px_per_sim_px() {
        // at 4x zoom one sim px is 4 screen px, so the projection shows 1/4 world unit
        // per screen px
        assert!((ortho_scale(4.0) - 0.25).abs() < 1e-6);
        assert!((ortho_scale(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn representation_crossfade_is_a_partition() {
        for z in [0.2, 0.4, 0.5, 0.75, 1.0, 6.0] {
            let sum = pixel_alpha(z) + schematic_alpha(z);
            assert!((sum - 1.0).abs() < 1e-6, "weights must sum to 1 at {z}");
        }
        assert_eq!(pixel_alpha(6.0), 1.0);
        assert_eq!(pixel_alpha(0.2), 0.0);
        assert_eq!(schematic_alpha(6.0), 0.0);
        // halfway through the fade band
        let mid = (PIXEL_FULL_ZOOM + PIXEL_GONE_ZOOM) / 2.0;
        assert!((pixel_alpha(mid) - 0.5).abs() < 1e-6);
    }

    /// Sculpting must be available across the whole snapped range, and unavailable exactly
    /// where the pixel grid stops being trustworthy.
    #[test]
    fn sculpt_threshold_sits_at_the_snap_boundary() {
        assert_eq!(SCULPT_MIN_ZOOM, 2.0);
        assert!(pixel_alpha(SCULPT_MIN_ZOOM) == 1.0);
    }
}
