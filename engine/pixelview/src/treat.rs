//! Per-layer treatment and the per-pixel readability rule. All CPU-side; no shaders.
//!
//! # Depth is a property of the slot, not of where the player is standing
//!
//! This module used to key everything off *relative* depth — `active()` for the layer you
//! were in, `behind(n)`/`front(n)` for the rest. That made the stack read as a set of
//! images you switch between: the same slab looked bright standing in it and murky
//! standing next to it, so the eye learned "bright == mine" rather than "small and hazy ==
//! far away".
//!
//! LittleBigPlanet's stack is lit and hazed by *distance*, and it looks the same
//! whichever slab Sackboy is on. So the treatment table is now absolute — indexed by slot
//! position front-to-back, [`LayerTreatment::at_depth`] — and being the active layer is a
//! light touch on top ([`LayerTreatment::emphasized`]), worth a little contrast and
//! nothing else. Geometry (`depth.rs`) carries the depth cue; this module only carries
//! *air*: haze, cool shift, desaturation, and the blur that stands in for depth of field.
//!
//! What survives unchanged from the vault:
//!
//! * "Hold-to-X-ray ghosts all layers with colour-coded outlines."  -> `xray()`, with
//!   `treatment_for` handing each depth its own hue so the ghosts are told apart.
//! * The reserved luma bands, so no palette can break the readability guarantee.
//!
//! What changed, and is changed in the vault to match:
//!
//! * "The camera never moves in Z" — it does now; see `depth.rs`.
//! * "Cavity layers are invisible until you open a panel" — a cavity you are not in is
//!   now a dark seam rather than nothing, because an invisible slot cannot show you that
//!   the slab in front of it is casting into it. `hidden()` and `hide_cavities` remain for
//!   anyone who wants the old behaviour.
//! * "Only the active layer gets full contrast" — the active layer gets a *little* more
//!   contrast. Its real distinction is that it is the only pixel-exact plane.

use pixelsim::LayerSlot;

#[derive(Copy, Clone, Debug)]
pub struct LayerTreatment {
    pub brightness: f32,
    pub saturation: f32,
    /// Depth-of-field blur. The geometric minimum for a resampled plane comes from
    /// [`crate::depth::shimmer_safe_blur_px`]; this is the artistic value, and the bake
    /// takes whichever is larger.
    pub blur_px: f32,
    pub alpha: f32,
    /// Hue rotation. Note what this cannot do: rotating the hue of a neutral colour returns
    /// the same colour, so on a scene made mostly of grey stone this axis is a no-op. Depth
    /// tinting is [`LayerTreatment::fog`]'s job for exactly that reason.
    pub hue_shift_deg: f32,
    /// The colour distance blends *toward* — aerial perspective, and the only depth cue that
    /// works on a material with no chroma of its own.
    pub fog: [f32; 3],
    /// How far toward `fog` this slot sits, 0..1.
    pub fog_amount: f32,
    /// front slab: 0.30   deepest: 0.00
    pub luma_lo: f32,
    /// front slab: 1.00   deepest: 0.45
    pub luma_hi: f32,
    pub half_res: bool,
    /// 60.0 for the two front slabs, 30.0 then 15.0 behind them
    pub bake_hz: f32,
    /// cavities, when `hide_cavities` is set
    pub hidden: bool,
}

/// How many slots deep the stack can be, and so how long the depth table is.
pub const MAX_DEPTH: usize = 6;

/// The colour of the air between the camera and a slab: a cool, slightly blue grey.
///
/// Warm-neutral would read as dust and make the vessel look dirty rather than deep. Cool
/// is what distance looks like, and it also pushes the rear slabs away from the warm
/// palette the near ones are painted in ([[Tone and Art Direction]]).
pub const FOG: [f32; 3] = [0.40, 0.46, 0.56];

/// How far the active layer is nudged toward looking one slot nearer than it is.
///
/// Strictly below 1.0, which is what keeps the nudge smaller than a slot of depth.
pub const EMPHASIS_FRACTION: f32 = 0.40;

impl LayerTreatment {
    /// The treatment for the slot `index` positions back from the front of the stack,
    /// standing in air of increasing thickness. Absolute: it does not know or care which
    /// layer the player is in.
    ///
    /// The axes, and why each is here rather than in the geometry:
    ///
    /// * **haze** (`brightness`, `luma_hi`) — air between you and the slab. Falls off
    ///   smoothly rather than in the old two-step jump, because a smooth ramp is what
    ///   lets six slots be told apart instead of just "near" and "far".
    /// * **cool shift** (`hue_shift_deg`) — aerial perspective. Distance is blue.
    /// * **desaturation** (`saturation`) — same reason, and it keeps a rear slab from
    ///   competing with a front one for the same hue.
    /// * **blur** (`blur_px`) — depth of field. It is also the low-pass a resampled plane
    ///   needs, so under perspective it is doing two jobs at once.
    /// * **bake rate** — motion pulls the eye harder than brightness does, so a distant
    ///   slab that is still churning per-pixel at 60 Hz reads as *near*. Rate falls with
    ///   depth for readability first and cost second.
    pub fn at_depth(index: usize) -> Self {
        // `t` = 0 at the front slab, 1 at the deepest
        let t = (index.min(MAX_DEPTH - 1) as f32) / (MAX_DEPTH - 1) as f32;
        let ease = t * t * 0.55 + t * 0.45; // slightly slow near the front, so slots 0-2 stay distinct
        Self {
            // Gentler than the flat renderer needed. Distance now reads from parallax,
            // apparent size, real occlusion and real shadow, so the haze ramp only has to
            // colour the air — not carry the whole cue on its own.
            brightness: 1.00 - 0.34 * ease,
            // Was 0.72, which drained nearly three quarters of the colour out of the back
            // of the stack. On a scene that is mostly stone — grey to begin with — that
            // plus the brightness ramp gave six shades of the same mud. Aerial perspective
            // is a *tint*, not a greyscale conversion; the vault's warmth has to survive
            // to the Hold.
            saturation: 1.00 - 0.42 * ease,
            blur_px: 2.2 * ease,
            alpha: 1.0,
            hue_shift_deg: -18.0 * ease,
            fog: FOG,
            // The axis that makes six slots of grey stone readable as six distances. Without
            // it the ramp had only brightness to work with on any neutral material, which is
            // most of a vessel dug out of rock.
            fog_amount: 0.30 * ease,
            luma_lo: 0.30 * (1.0 - ease),
            luma_hi: 1.00 - 0.55 * ease,
            half_res: index >= 2,
            // Measured, not guessed. The first version of this table gave slot 1 60 Hz at
            // full resolution — four times the rate and twice the resolution of the rear
            // treatment it replaced — which on its own cost more than both slab passes
            // together. The active layer is separately pinned to 60 Hz by `emphasized`, so
            // whichever slab the player is in stays smooth regardless of what this says.
            bake_hz: match index {
                0 => 60.0,
                1 => 30.0,
                2 | 3 => 20.0,
                _ => 12.0,
            },
            hidden: false,
        }
    }

    /// The layer the player is in, as a nudge on top of its depth treatment.
    ///
    /// The nudge is expressed as a *fraction of one slot of depth*: `nearer` is the
    /// treatment this slot would have if it sat one slot closer, and the active layer moves
    /// [`EMPHASIS_FRACTION`] of the way toward it. That construction is the whole point. An
    /// absolute nudge — the first version of this was "18% brighter" — can exceed a slot of
    /// depth, which puts an active Works slab ahead of the Deck in front of it and restores
    /// exactly the bright-means-mine reading the absolute table exists to kill. As a
    /// fraction of a slot it cannot, by construction, and the test that caught it
    /// (`the_active_nudge_is_smaller_than_a_slot_of_depth`) now guards it.
    ///
    /// Focus is the exception, and is not a fraction of anything: the active plane goes to
    /// zero blur and full bake rate. Depth of field follows the player in LBP too, and this
    /// is the one plane drawn at `ratio == 1.0`, so it is the only one with no resampling
    /// to hide.
    ///
    /// `k` is how much of the nudge to apply, which is what the 0.3 s switch animates.
    pub fn emphasized(self, k: f32, nearer: Self) -> Self {
        let k = k.clamp(0.0, 1.0) * EMPHASIS_FRACTION;
        let mut t = Self::lerp(self, nearer, k);
        t.blur_px = 0.0;
        t.half_res = false;
        t.bake_hz = self.bake_hz.max(60.0);
        t.alpha = self.alpha;
        t
    }

    /// A slab in front of the one you are working in gets a little translucent so it
    /// cannot wall you out — LBP leans on the camera for this, we cannot, and a fully
    /// opaque Face slab makes the Deck unworkable.
    pub fn see_through(self, alpha: f32) -> Self {
        Self { alpha, ..self }
    }

    /// Cavities, when `hide_cavities` is set. `bake_hz = 0.0` means never baked at all —
    /// invisible content should cost nothing, not merely draw nothing.
    pub fn hidden() -> Self {
        Self {
            brightness: 0.0,
            saturation: 0.0,
            blur_px: 0.0,
            alpha: 0.0,
            hue_shift_deg: 0.0,
            fog: FOG,
            fog_amount: 0.0,
            luma_lo: 0.00,
            luma_hi: 0.00,
            half_res: true,
            bake_hz: 0.0,
            hidden: true,
        }
    }

    /// 0.25 / 0.15 / no blur, outline on.
    ///
    /// X-ray ghosts *every* layer, so `luma_hi` sits above the 0.35 rear ceiling — while
    /// the key is held the depth-ordering rule is deliberately suspended in favour of
    /// seeing everything at once. `hue_shift_deg` is overwritten per depth by
    /// [`RenderConfig::treatment_for`], which is what makes the outlines colour-coded.
    pub fn xray() -> Self {
        Self {
            brightness: 0.25,
            saturation: 0.15,
            blur_px: 0.0,
            alpha: 1.0,
            hue_shift_deg: 0.0,
            fog: FOG,
            fog_amount: 0.0,
            luma_lo: 0.05,
            luma_hi: 0.55,
            half_res: false,
            bake_hz: 30.0,
            hidden: false,
        }
    }

    /// Cross-animate two treatments — the 0.3 s layer switch. It now animates only the
    /// active-layer nudge, since depth itself no longer changes when you shift: the slab
    /// you left keeps the haze its distance earns.
    ///
    /// The booleans do not lerp: `half_res` takes the crisper of the two so a layer
    /// becoming active sharpens immediately, `bake_hz` takes the faster so nothing
    /// under-samples mid-animation, and `hidden` survives only if *both* are hidden — so
    /// revealing a cavity fades in through `alpha` instead of popping.
    pub fn lerp(a: Self, b: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let f = |x: f32, y: f32| x + (y - x) * t;
        Self {
            brightness: f(a.brightness, b.brightness),
            saturation: f(a.saturation, b.saturation),
            blur_px: f(a.blur_px, b.blur_px),
            alpha: f(a.alpha, b.alpha),
            hue_shift_deg: f(a.hue_shift_deg, b.hue_shift_deg),
            fog: [
                f(a.fog[0], b.fog[0]),
                f(a.fog[1], b.fog[1]),
                f(a.fog[2], b.fog[2]),
            ],
            fog_amount: f(a.fog_amount, b.fog_amount),
            luma_lo: f(a.luma_lo, b.luma_lo),
            luma_hi: f(a.luma_hi, b.luma_hi),
            half_res: a.half_res && b.half_res,
            bake_hz: a.bake_hz.max(b.bake_hz),
            hidden: a.hidden && b.hidden,
        }
    }
}

/// The baked slab lip: how a flat texture pretends to have thickness.
///
/// LBP's slabs are real extrusions and you see their side faces. We cannot show a plane
/// at an angle (see `depth.rs`), so the face is drawn into the layer's own texture as a
/// lip along one side of every silhouette edge — which is what an LBP face looks like at
/// our depths anyway.
#[derive(Clone, Copy, Debug)]
pub struct ExtrudeConfig {
    /// Lip depth in cells. 0 disables the pass. Thin slots get half of this.
    pub px: f32,
    /// Direction the lip is thrown, as a unit-ish vector in cell space (`+y` is down).
    /// Fixed rather than derived per-pixel from the camera, because the bake must not
    /// depend on where the camera is — it would invalidate every tile on every pan.
    pub dir: (f32, f32),
    /// How much darker the lip is than the face it came from.
    pub shade: f32,
}

impl Default for ExtrudeConfig {
    fn default() -> Self {
        Self {
            // Off. Slabs have real thickness now (`solid.rs`), so a painted lip would be a
            // second, disagreeing edge drawn on top of the real one. Kept switchable because
            // it is a useful A/B for how much of the depth read is geometry.
            px: 0.0,
            // down and to the right: light comes from the upper left, as it does in every
            // Scarry cutaway ever drawn
            dir: (0.55, 0.83),
            shade: 0.45,
        }
    }
}

/// The cue that glues the planes into one space: a slab casts onto the slab behind it.
#[derive(Clone, Copy, Debug)]
pub struct ShadowConfig {
    /// How dark the deepest part of a cast shadow gets. 0 disables the pass.
    pub strength: f32,
    /// Cells the shadow is displaced by, per slot of depth between caster and receiver —
    /// same light direction as [`ExtrudeConfig::dir`].
    pub offset_px: f32,
    /// Softening radius of the cast, in coarse mask cells.
    pub softness: u8,
    /// Extra darkening where the caster is directly in front, giving the contact shading
    /// that reads as two surfaces touching rather than two images stacked.
    pub contact: f32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            // Off, for the same reason: the sun casts real shadows through real geometry, and
            // a baked cast on top of that double-darkens whatever it lands on.
            strength: 0.0,
            offset_px: 6.0,
            softness: 2,
            contact: 0.25,
        }
    }
}

#[derive(bevy::prelude::Resource, Clone, Debug)]
pub struct RenderConfig {
    /// Absolute depth table, front slot to back.
    pub by_depth: [LayerTreatment; MAX_DEPTH],
    pub xray: LayerTreatment,
    /// How much of the active-layer nudge to apply — see [`LayerTreatment::emphasized`].
    pub active_emphasis: f32,
    /// Alpha for slabs in front of the active one, nearest first. They have to thin out or
    /// they wall the player out of their own workspace.
    pub front_alpha: [f32; MAX_DEPTH],
    /// The old behaviour: cavities invisible unless you are in them.
    pub hide_cavities: bool,
    pub transition_secs: f32,
    /// dithered hole around the player in front layers
    pub punch_through_px: f32,
    /// debug: dead-zone visualisation
    pub tint_sleeping: bool,
    pub extrude: ExtrudeConfig,
    pub shadow: ShadowConfig,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            by_depth: [
                LayerTreatment::at_depth(0),
                LayerTreatment::at_depth(1),
                LayerTreatment::at_depth(2),
                LayerTreatment::at_depth(3),
                LayerTreatment::at_depth(4),
                LayerTreatment::at_depth(5),
            ],
            xray: LayerTreatment::xray(),
            active_emphasis: 1.0,
            // one slab in front is a scrim you see through; two is nearly gone
            front_alpha: [0.55, 0.30, 0.18, 0.12, 0.08, 0.06],
            hide_cavities: false,
            transition_secs: 0.30,
            punch_through_px: 24.0,
            tint_sleeping: false,
            extrude: ExtrudeConfig::default(),
            shadow: ShadowConfig::default(),
        }
    }
}

/// hue degrees between adjacent layers under x-ray, so the ghosts are colour-coded
pub const XRAY_HUE_STEP_DEG: f32 = 40.0;

impl RenderConfig {
    /// The treatment for the layer at absolute stack position `index` (0 == front-most),
    /// given how far it sits from the active layer.
    ///
    /// `depth` is signed exactly as `Sim::depth_from` produces it: positive behind the
    /// active layer, negative in front, zero for the active layer. It no longer chooses
    /// the treatment — `index` does — and is used only for the three things that genuinely
    /// depend on where the player is: the active-layer nudge, the see-through scrim in
    /// front of them, and the x-ray hue ramp.
    pub fn treatment_for(
        &self,
        index: usize,
        depth: i32,
        slot: LayerSlot,
        xray: bool,
    ) -> LayerTreatment {
        if xray {
            let mut t = self.xray;
            // colour-coded outlines, per the vault's x-ray rule: each depth gets its own
            // hue so ghosted layers are distinguishable by colour, not by brightness
            // (which x-ray has flattened on purpose)
            t.hue_shift_deg = depth as f32 * XRAY_HUE_STEP_DEG;
            if depth == 0 {
                // you still own the layer you are standing in, even under x-ray
                t.luma_lo = 0.30;
                t.luma_hi = 0.85;
            }
            return t;
        }
        if depth != 0 && self.hide_cavities && matches!(slot, LayerSlot::Cavity | LayerSlot::Cavity)
        {
            return LayerTreatment::hidden();
        }

        let index = index.min(MAX_DEPTH - 1);
        let base = self.by_depth[index];
        if depth == 0 {
            // slot 0 has nothing in front of it, so it is its own reference and the nudge
            // reduces to focus alone
            let nearer = self.by_depth[index.saturating_sub(1)];
            return base.emphasized(self.active_emphasis, nearer);
        }
        if depth < 0 {
            // in front of the player: thin it out. Nearest first, so `-1` picks [0].
            let steps = depth.unsigned_abs().clamp(1, MAX_DEPTH as u32) as usize;
            return base.see_through(self.front_alpha[steps - 1]);
        }
        base
    }
}

/// The readability rule, applied per pixel: brightness -> desaturate -> hue shift
/// -> luma-range compression.
///
/// Correct, but not the hot path — it rebuilds the colour matrix on every call. `bake`
/// uses [`PixelTreat`], which hoists that out of the loop; this function is defined in
/// terms of the same code so the two cannot drift.
#[inline]
pub fn treat_pixel(rgb: [u8; 3], t: &LayerTreatment) -> [u8; 3] {
    PixelTreat::new(t).apply(rgb)
}

/// Rec. 709 luma: 0.2126 / 0.7152 / 0.0722
#[inline]
pub fn luma(rgb: [u8; 3]) -> f32 {
    let [r, g, b] = rgb;
    (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32) / 255.0
}

const LR: f32 = 0.2126;
const LG: f32 = 0.7152;
const LB: f32 = 0.0722;

/// The luma window a treatment can emit, `(lo, hi)`.
#[inline]
pub fn luma_range(t: &LayerTreatment) -> (f32, f32) {
    (t.luma_lo, t.luma_hi)
}

/// WCAG-style contrast ratio between the darkest pixel `a` can emit and the brightest
/// pixel `b` can emit — the *worst case* at a silhouette edge between two layers.
///
/// Exposed so the legibility spike can measure the readability claim rather than assert
/// it. Note what it returns for the shipped defaults: active `luma_lo` 0.40 against rear
/// `luma_hi` 0.35 is 1.11:1, **not** the 3:1 the plan claims. The 3:1 figure holds only
/// between the *means* of the two bands (0.70 vs 0.175, about 3.4:1).
#[inline]
pub fn worst_case_contrast(a: &LayerTreatment, b: &LayerTreatment) -> f32 {
    let lo = a.luma_lo;
    let hi = b.luma_hi;
    (lo.max(hi) + 0.05) / (lo.min(hi) + 0.05)
}

/// A [`LayerTreatment`] with brightness, saturation and hue folded into one 3x3 matrix.
///
/// Built once per bake. This exists because the hue rotation needs a `sin`/`cos` pair:
/// evaluated per pixel that is ~1.5 M transcendental pairs per frame across three
/// 1024x512 layers, which alone would blow the 60 fps budget. Hoisted, the whole
/// per-pixel treatment is 9 multiplies, 6 adds, a luma dot product and one scale.
#[derive(Copy, Clone, Debug)]
pub struct PixelTreat {
    m: [f32; 9],
    luma_lo: f32,
    luma_span: f32,
    /// premultiplied fog target, 0..255 per channel
    fog: [f32; 3],
    fog_k: f32,
}

impl PixelTreat {
    pub fn new(t: &LayerTreatment) -> Self {
        // saturation: S = s*I + (1-s)*L, where every row of L is the luma vector
        let s = t.saturation;
        let is = 1.0 - s;
        let sat = [
            s + is * LR,
            is * LG,
            is * LB,
            is * LR,
            s + is * LG,
            is * LB,
            is * LR,
            is * LG,
            s + is * LB,
        ];

        // luma-preserving hue rotation about the grey axis (the standard matrix)
        let (sa, ca) = t.hue_shift_deg.to_radians().sin_cos();
        let hue = [
            0.213 + ca * 0.787 - sa * 0.213,
            0.715 - ca * 0.715 - sa * 0.715,
            0.072 - ca * 0.072 + sa * 0.928,
            0.213 - ca * 0.213 + sa * 0.143,
            0.715 + ca * 0.285 + sa * 0.140,
            0.072 - ca * 0.072 - sa * 0.283,
            0.213 - ca * 0.213 - sa * 0.787,
            0.715 - ca * 0.715 + sa * 0.715,
            0.072 + ca * 0.928 + sa * 0.072,
        ];

        // brightness is a scalar, so it folds straight into the product: M = H * S * b
        let b = t.brightness;
        let mut m = [0.0f32; 9];
        for r in 0..3 {
            for c in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += hue[r * 3 + k] * sat[k * 3 + c];
                }
                m[r * 3 + c] = acc * b;
            }
        }

        Self {
            m,
            luma_lo: t.luma_lo,
            luma_span: t.luma_hi - t.luma_lo,
            fog: [t.fog[0] * 255.0, t.fog[1] * 255.0, t.fog[2] * 255.0],
            fog_k: t.fog_amount.clamp(0.0, 1.0),
        }
    }

    /// The last step is the luma reservation, and it is unconditional:
    /// `v_out = luma_lo + v_in * (luma_hi - luma_lo)`, applied as a scale on the RGB
    /// triple about its own luma, falling back to compression toward white when the
    /// colour has no headroom left to scale into. That is what keeps the active layer and
    /// the rear layers in disjoint luma bands for *any* palette, present or future — see
    /// the headroom note in the body for why the plain scale is not enough on its own.
    #[inline]
    pub fn apply(&self, rgb: [u8; 3]) -> [u8; 3] {
        let r = rgb[0] as f32;
        let g = rgb[1] as f32;
        let b = rgb[2] as f32;
        let m = &self.m;
        let mut out = [
            m[0] * r + m[1] * g + m[2] * b,
            m[3] * r + m[4] * g + m[5] * b,
            m[6] * r + m[7] * g + m[8] * b,
        ];

        // Aerial perspective: blend toward the fog colour before the luma reservation, so a
        // distant slab picks up the colour of the air it is behind even when the material
        // itself has no chroma to shift. This is the axis `hue_shift_deg` cannot provide —
        // see the note on that field.
        if self.fog_k > 0.0 {
            let k = self.fog_k;
            out[0] += (self.fog[0] - out[0]) * k;
            out[1] += (self.fog[1] - out[1]) * k;
            out[2] += (self.fog[2] - out[2]) * k;
        }

        let v = (LR * out[0] + LG * out[1] + LB * out[2]) / 255.0;
        let target = self.luma_lo + v.clamp(0.0, 1.0) * self.luma_span;

        if v <= 1.0e-4 {
            // a black pixel has no hue to preserve; lift it to the reserved floor as grey
            let grey = target * 255.0;
            out = [grey, grey, grey];
        } else {
            let k = target / v;
            let peak = out[0].max(out[1]).max(out[2]);
            // Headroom check. A plain scale cannot brighten a colour whose brightest
            // channel is already at 255 — pure red scaled by 2.5 is still pure red once
            // clamped, and its luma stays at 0.21, well below the active band's 0.40
            // floor. Left alone that silently voids the whole readability guarantee for
            // saturated palette entries. So: scale as far as the headroom allows, then
            // make up the remaining luma deficit by compressing toward white, which is
            // the minimum loss of chroma that still lands in the band. Hue is preserved
            // either way; saturation is what gets spent, and the vault is explicit that
            // readability outranks it.
            let k_max = if peak > 0.0 { 255.0 / peak } else { k };
            if k <= k_max {
                out[0] *= k;
                out[1] *= k;
                out[2] *= k;
            } else {
                out[0] *= k_max;
                out[1] *= k_max;
                out[2] *= k_max;
                let reached = v * k_max;
                let headroom = 1.0 - reached;
                if headroom > 1.0e-4 {
                    let t = ((target - reached) / headroom).clamp(0.0, 1.0);
                    out[0] += (255.0 - out[0]) * t;
                    out[1] += (255.0 - out[1]) * t;
                    out[2] += (255.0 - out[2]) * t;
                }
            }
        }

        [
            out[0].clamp(0.0, 255.0) as u8,
            out[1].clamp(0.0, 255.0) as u8,
            out[2].clamp(0.0, 255.0) as u8,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spread of awkward colours: pure primaries, extremes, and the actual slice
    /// palette entries (sand, water, bedrock, gold).
    const SAMPLES: [[u8; 3]; 10] = [
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [194, 178, 128],
        [58, 110, 165],
        [60, 60, 60],
        [212, 175, 55],
        [1, 0, 2],
    ];

    /// The readability rule is only worth anything if it is *unconditional*: every pixel
    /// a treatment can emit must land inside that treatment's reserved luma band, for any
    /// input colour. This is the test that lets a future palette be added safely.
    #[test]
    fn luma_always_lands_in_the_reserved_band() {
        let mut treatments: Vec<LayerTreatment> =
            (0..MAX_DEPTH).map(LayerTreatment::at_depth).collect();
        treatments.push(LayerTreatment::at_depth(1).emphasized(1.0, LayerTreatment::at_depth(0)));
        treatments.push(LayerTreatment::xray());
        for t in treatments {
            for rgb in SAMPLES {
                let v = luma(treat_pixel(rgb, &t));
                // one 8-bit quantisation step of slack
                let eps = 1.5 / 255.0;
                assert!(
                    v >= t.luma_lo - eps && v <= t.luma_hi + eps,
                    "luma {v} escaped [{}, {}] for {rgb:?}",
                    t.luma_lo,
                    t.luma_hi
                );
            }
        }
    }

    /// The bands no longer have to be *disjoint* — depth is carried by geometry now, and
    /// disjoint bands were what forced the old two-step "near or far" look. What must hold
    /// is that every axis moves monotonically with depth, by a margin big enough to see:
    /// six slots have to be distinguishable from each other, not merely from the active one.
    #[test]
    fn every_depth_axis_is_monotonic_with_a_visible_margin() {
        let table: Vec<LayerTreatment> = (0..MAX_DEPTH).map(LayerTreatment::at_depth).collect();
        for w in table.windows(2) {
            let (near, far) = (w[0], w[1]);
            assert!(far.brightness < near.brightness, "haze must deepen");
            assert!(far.saturation < near.saturation, "colour must drain");
            assert!(far.blur_px > near.blur_px, "focus must fall off");
            assert!(far.fog_amount > near.fog_amount, "distance must haze");
            assert!(far.luma_hi < near.luma_hi, "the ceiling must come down");
            assert!(
                near.luma_hi - far.luma_hi > 0.04,
                "adjacent slots differ by only {}",
                near.luma_hi - far.luma_hi
            );
            assert!(
                far.bake_hz <= near.bake_hz,
                "motion must not increase with depth"
            );
        }
    }

    /// The active-layer nudge is meant to be a nudge. If it grows enough to outrank one
    /// whole slot of depth, the stack goes back to reading as focus state.
    #[test]
    fn the_active_nudge_is_smaller_than_a_slot_of_depth() {
        for i in 1..MAX_DEPTH {
            let plain = LayerTreatment::at_depth(i);
            let nearer = LayerTreatment::at_depth(i - 1);
            let active = plain.emphasized(1.0, nearer);
            assert!(
                active.brightness <= nearer.brightness + 1e-6,
                "slot {i} emphasized ({}) outshines slot {} ({})",
                active.brightness,
                i - 1,
                nearer.brightness
            );
            // and it does something, or it is not worth having
            assert!(active.luma_hi > plain.luma_hi || active.brightness > plain.brightness);
            // strictly inside the slot it is in, never past the slot in front
            assert!(active.luma_hi < nearer.luma_hi + 1e-6);
            // the plane in focus is sharp whatever its depth says
            assert_eq!(active.blur_px, 0.0);
        }
    }

    /// Pins the measured contrast at a silhouette edge between neighbouring slots, so the
    /// legibility spike's numbers can be compared against something.
    #[test]
    fn adjacent_slot_contrast_is_pinned() {
        let r = worst_case_contrast(&LayerTreatment::at_depth(0), &LayerTreatment::at_depth(1));
        assert!(r > 1.0, "neighbouring slots must not be identical");
        assert!(r < 3.0, "if this passes 3:1 the ramp was retuned");
    }

    /// Regression: a fully saturated primary has no headroom to scale into, so a naive
    /// "scale about its own luma" clamps and leaves the pixel below the active band's
    /// floor. It must be compressed toward white instead, keeping its hue.
    #[test]
    fn saturated_primaries_are_lifted_into_the_active_band() {
        let t = LayerTreatment::at_depth(0);
        for rgb in [[255u8, 0, 0], [0, 0, 255], [0, 255, 0]] {
            let out = treat_pixel(rgb, &t);
            assert!(
                luma(out) >= t.luma_lo - 1.5 / 255.0,
                "{rgb:?} -> {out:?} has luma {} below the {} floor",
                luma(out),
                t.luma_lo
            );
            // hue survives: the originally dominant channel is still the brightest
            let dominant = (0..3).max_by_key(|&i| rgb[i]).unwrap();
            assert!(
                out[dominant] >= out[(dominant + 1) % 3]
                    && out[dominant] >= out[(dominant + 2) % 3],
                "hue was not preserved: {rgb:?} -> {out:?}"
            );
        }
    }

    #[test]
    fn saturation_zero_is_grey() {
        let mut t = LayerTreatment::at_depth(0);
        t.saturation = 0.0;
        t.hue_shift_deg = 0.0;
        let out = treat_pixel([200, 40, 90], &t);
        assert!(
            out[0].abs_diff(out[1]) <= 2 && out[1].abs_diff(out[2]) <= 2,
            "expected grey, got {out:?}"
        );
    }

    /// A hue rotation must move hue without moving brightness — otherwise the depth cue
    /// and the luma reservation fight each other.
    #[test]
    fn hue_shift_preserves_luma() {
        let mut t = LayerTreatment::at_depth(0);
        t.luma_lo = 0.0;
        t.luma_hi = 1.0;
        let base = luma(treat_pixel([180, 90, 40], &t));
        for deg in [-24.0, -12.0, 8.0, 40.0, 120.0] {
            t.hue_shift_deg = deg;
            let v = luma(treat_pixel([180, 90, 40], &t));
            assert!(
                (v - base).abs() < 0.02,
                "luma moved by {} at {deg}",
                v - base
            );
        }
    }

    #[test]
    fn lerp_hits_both_endpoints() {
        let a = LayerTreatment::at_depth(0);
        let b = LayerTreatment::at_depth(2);
        let at0 = LayerTreatment::lerp(a, b, 0.0);
        let at1 = LayerTreatment::lerp(a, b, 1.0);
        assert!((at0.brightness - a.brightness).abs() < 1e-6);
        assert!((at1.brightness - b.brightness).abs() < 1e-6);
        assert!((at1.luma_hi - b.luma_hi).abs() < 1e-6);
    }

    /// Revealing a cavity must fade rather than pop, so `hidden` may not survive a lerp
    /// that has a visible endpoint.
    #[test]
    fn lerp_out_of_hidden_fades_instead_of_popping() {
        let mid = LayerTreatment::lerp(LayerTreatment::hidden(), LayerTreatment::at_depth(0), 0.5);
        assert!(!mid.hidden);
        assert!(mid.alpha > 0.0 && mid.alpha < 1.0);
    }

    #[test]
    fn treat_pixel_matches_the_hoisted_path() {
        let t = LayerTreatment::at_depth(1);
        let pt = PixelTreat::new(&t);
        for rgb in SAMPLES {
            assert_eq!(treat_pixel(rgb, &t), pt.apply(rgb));
        }
    }
}
