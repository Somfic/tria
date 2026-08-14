//! CPU-side baking of cell arrays into an RGBA8 `Image`. Dirty-chunk tiles only,
//! unless `force_full`.
//!
//! # Four deliberate departures from a naive per-frame full rebake
//!
//! **Dirty tiles.** The solver marks chunks dirty on any write; only those 64x64 tiles
//! are rewritten. A settled vessel bakes nothing. This holds for *every* treatment,
//! blurred rear layers included — see below.
//!
//! **`half_res` is a sampling stride, not a smaller texture.** Every canvas allocates at
//! full layer resolution and `half_res` makes the bake sample one cell in four and write
//! 2x2 blocks. That keeps the intended 4x saving in *bake cost* — the cost that actually
//! mattered — while avoiding a texture reallocation every time the player switches
//! layers, and keeping the active layer pixel-exact at all times. In Bevy 0.18 resizing
//! an `Image` drops the GPU texture and forces a full re-upload, so a realloc
//! mid-transition is the one thing worth designing out. The softness a linear upscale
//! would have provided is delivered by [`box_blur_rgba`] instead.
//!
//! **Blur runs out of a shadow buffer, tile-wise, separably.** A blur reads neighbouring
//! pixels, so blurring a tile in place would seam at tile edges *and* compound on every
//! rebake (blurring already-blurred output). The fix is not to give up on dirty tiles but
//! to keep the unblurred bake in a second buffer (`sharp`) and blur *out of* it into the
//! texture: reads never touch blurred output, so nothing compounds, and reading a 1px
//! apron out of `sharp` — which is valid everywhere, not just inside the dirty tile —
//! means nothing seams. The blur itself is separable (two 3-tap passes rather than one
//! 9-tap window) and accumulates into a three-row ring instead of cloning the whole
//! texture. All of it is bit-exact against the old 9-tap; see
//! `blur_matches_the_naive_window` and `tile_blur_matches_a_whole_image_blur`.
//!
//! **Uploads are proportional to the dirty area.** Bevy 0.18 has no partial image upload,
//! but the CPU-side copy into `Image::data` is ours to control, so
//! [`LayerCanvas::upload`] copies only the rows the bake touched.
//!
//! # Any change of treatment forces a full rebake
//!
//! Dirty tiles express "these cells changed". They cannot express "every pixel's colour
//! changed because the layer is now dimmer, or ghosted, or blurred". So the bake
//! fingerprints the treatment and rebakes whole whenever the fingerprint moves — which is
//! what makes toggling x-ray, starting a layer transition, or flipping the sleeping-cell
//! tint correct rather than merely cheap. Callers do not have to remember to ask.
//!
//! # What it costs, measured
//!
//! Release build, 1024x512 layers, three of them, the shipping [`crate::RenderConfig`].
//! Per `FixedUpdate` tick, against a 16.6 ms budget:
//!
//! * settled or lightly churning (eight dirty tiles per layer): **0.66 ms**
//! * one dirty tile, one layer, bake and upload: **0.023 ms**
//! * a blurred rear layer with nothing dirty: **0.13 ms** (it was 12.4 ms, whole-layer,
//!   every tick)
//! * every layer rebaking whole — the tick an x-ray toggle lands on, and each of the
//!   ~18 ticks of the 0.3 s layer switch: **5.7 ms**
//!
//! So the expensive shape is real and is still about a third of a tick, but it is now
//! confined to ticks on which the *treatment* changed — exactly the ticks the player asked
//! for. It is not, as an earlier version of this comment claimed, something the rear-layer
//! bake rate makes free — a whole-layer blur is 4.7 ms of that on its own and no `bake_hz`
//! hides it. What makes it affordable is that a settled frame does not do it at all.

use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::image::{Image, ImageSampler, ImageSamplerDescriptor};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use pixelsim::{CHUNK_PX, FLAG_SLEEPING, Layer, MaterialClass, MaterialTable};

use crate::palette::{Palette, grain_jitter, grain_of, speckle_byte};
use crate::schematic::SchematicCache;
use crate::slab::{MASK_DIV, SlabMask, lip_at};
use crate::treat::{ExtrudeConfig, LayerTreatment, PixelTreat, ShadowConfig};

/// blur is only worth its cost from this radius up; below it the half-res stride and the
/// treatment's own dimming carry the depth cue
pub const BLUR_MIN_PX: f32 = 2.0;
/// how much brighter a silhouette edge is drawn under x-ray
pub const OUTLINE_BOOST: f32 = 1.9;
/// how transparent the *interior* of a ghosted layer is under x-ray
pub const XRAY_FILL_ALPHA: u8 = 64;
/// how far a cast shadow leans toward the fog colour, as a fraction of its own darkness
pub const SHADOW_TINT: f32 = 0.22;

/// Inclusive cell rect, `(x0, y0, x1, y1)`.
pub(crate) type Rect = (u16, u16, u16, u16);

pub struct LayerCanvas {
    pub handle: Handle<Image>,
    /// texture size (always the layer size — `half_res` is a stride, not a resize)
    pub w: u16,
    pub h: u16,
    pub rgba: Vec<u8>,
    pub half_res: bool,
    pub force_full: bool,
    /// seconds since the last bake, against `treatment.bake_hz`
    pub accum: f32,
    /// The unblurred bake, kept only for blurred treatments — the blur's source of truth,
    /// so a tile blur can read a valid apron and never compounds. Empty otherwise.
    sharp: Vec<u8>,
    /// fingerprint of the last bake's treatment; a change means every pixel changed
    sig: Option<BakeSig>,
    /// memoised grain and treated colour, valid for one `sig`
    cache: BakeCache,
    /// three-row horizontal-pass ring for the separable blur, hoisted out of the loop
    ring: Vec<u32>,
    /// Coarse occupancy of this layer, so the layer *behind* it can be cast onto. Only
    /// filled when someone asks for it via [`LayerCanvas::update_mask`].
    pub(crate) mask: SlabMask,
    /// what the sim dirtied on the last bake, un-grown — the caster footprint the layer
    /// behind needs
    last_dirty: Vec<Rect>,
    /// rects invalidated from outside, e.g. by a caster in front moving
    extra: Vec<Rect>,
    /// the rects this bake is writing, chunk-aligned and deduped
    touched: Vec<Rect>,
    /// per-chunk "already queued" flags for that dedupe, reused across bakes
    touch_set: Vec<bool>,
    /// what the GPU image still owes
    upload: Upload,
    /// coarse cache for [`crate::bake_schematic`]; only the schematic canvas fills it
    pub(crate) schematic: SchematicCache,
}

/// The pixels [`LayerCanvas::upload`] still has to hand the `Image`.
#[derive(Default)]
struct Upload {
    all: bool,
    rects: Vec<Rect>,
    /// pixels covered by `rects`, to decide when one memcpy beats many row copies
    area: usize,
}

impl Upload {
    fn nothing(&mut self) {
        self.all = false;
        self.rects.clear();
        self.area = 0;
    }

    fn everything(&mut self) {
        self.all = true;
        self.rects.clear();
        self.area = 0;
    }

    fn add(&mut self, r: Rect, canvas_area: usize) {
        if self.all {
            return;
        }
        self.area += (r.2 - r.0 + 1) as usize * (r.3 - r.1 + 1) as usize;
        // past half the canvas the row-by-row copy loses to a single memcpy and the rect
        // list stops being worth walking
        if self.area * 2 >= canvas_area {
            self.everything();
        } else {
            self.rects.push(r);
        }
    }

    fn pending(&self) -> bool {
        self.all || !self.rects.is_empty()
    }
}

/// Everything one bake needs that is not the layer itself.
///
/// `extrude` and `cast` are the slab cues; both are `None` for the schematic canvas and
/// for [`LayerCanvas::bake_with`], which is what keeps the plain bake path unchanged.
pub struct BakeCtx<'a> {
    pub table: &'a MaterialTable,
    pub pal: &'a Palette,
    pub treat: &'a LayerTreatment,
    pub tint_sleeping: bool,
    pub outline: bool,
    /// `(config, depth_in_cells)` — depth is per-slot, so a thin cavity gets a shallower
    /// lip than a thick slab without needing its own config.
    pub extrude: Option<(ExtrudeConfig, f32)>,
    /// What the slab in front throws onto this one.
    pub cast: Option<Cast<'a>>,
    /// Geometric blur floor for a resampled plane, from [`crate::depth::shimmer_safe_blur_px`].
    pub min_blur_px: f32,
}

impl BakeCtx<'_> {
    /// How far outside one of *this* layer's changed cells the bake can write, in cells.
    ///
    /// Only the lip reaches: a changed cell throws a lip up to `depth` cells away. The cast
    /// does *not* belong here even though it is bigger, because on the receiving side a
    /// cast is a per-pixel multiply — it has no spatial spread at all. Conflating the two
    /// grew every dirty tile by 18 cells instead of 3, which is a 2.4x area increase on
    /// every tile the sim touched, and was most of a 6x bake regression.
    pub fn lip_reach(&self) -> u16 {
        let lip = self.extrude.map(|(_, d)| d.ceil()).unwrap_or(0.0);
        (lip as u16).min(64)
    }

    /// How far a *caster's* changed cell reaches into this layer, in cells: the shadow
    /// displacement plus the softening radius. This is what the caster's invalidated rects
    /// are grown by, and it applies to nothing else.
    pub fn cast_reach(&self) -> u16 {
        let cast = self.cast.as_ref().map(|c| c.reach_px()).unwrap_or(0.0);
        (cast.ceil() as u16).min(128)
    }
}

/// One slab's shadow, as the layer behind it sees it.
pub struct Cast<'a> {
    pub mask: &'a SlabMask,
    pub cfg: ShadowConfig,
    /// cell-space displacement of the cast, from [`crate::slab::cast_offset`]
    pub offset: (f32, f32),
}

impl Cast<'_> {
    fn strength(&self) -> f32 {
        if self.mask.is_empty() {
            0.0
        } else {
            self.cfg.strength
        }
    }

    fn reach_px(&self) -> f32 {
        let (ox, oy) = self.offset;
        ox.abs().max(oy.abs()) + (self.cfg.softness as f32 + 1.0) * MASK_DIV as f32
    }
}

/// Everything about a treatment that changes baked pixels, as a comparable value.
///
/// `alpha` and `parallax` are absent on purpose: they are sprite properties applied at
/// draw time, so moving them must *not* trigger a rebake. `bake_hz` gates when the bake
/// runs rather than what it produces, and `hidden` returns before any of this.
#[derive(Clone, Copy, PartialEq, Eq)]
struct BakeSig {
    brightness: u32,
    saturation: u32,
    /// the *decision*, not `blur_px`: the kernel is a fixed 3x3, so moving `blur_px`
    /// without crossing `BLUR_MIN_PX` changes no pixel and must not cost a rebake
    blur: bool,
    hue_shift_deg: u32,
    fog_amount: u32,
    luma_lo: u32,
    luma_hi: u32,
    half_res: bool,
    outline: bool,
    tint_sleeping: bool,
    /// lip depth and cast strength change every pixel too
    lip: u32,
    cast: u32,
}

impl BakeSig {
    #[allow(clippy::too_many_arguments)]
    fn new(
        t: &LayerTreatment,
        half_res: bool,
        outline: bool,
        tint_sleeping: bool,
        lip: f32,
        cast: f32,
        blur: bool,
    ) -> Self {
        Self {
            brightness: t.brightness.to_bits(),
            saturation: t.saturation.to_bits(),
            blur,
            hue_shift_deg: t.hue_shift_deg.to_bits(),
            fog_amount: t.fog_amount.to_bits(),
            luma_lo: t.luma_lo.to_bits(),
            luma_hi: t.luma_hi.to_bits(),
            half_res,
            outline,
            tint_sleeping,
            lip: lip.to_bits(),
            cast: cast.to_bits(),
        }
    }
}

impl LayerCanvas {
    pub fn new(images: &mut Assets<Image>, w: u16, h: u16, half_res: bool, nearest: bool) -> Self {
        let rgba = vec![0u8; w as usize * h as usize * 4];
        let mut image = Image::new(
            Extent3d {
                width: w as u32,
                height: h as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            rgba.clone(),
            TextureFormat::Rgba8UnormSrgb,
            // keep the CPU buffer authoritative: MAIN_WORLD means the extractor clones
            // rather than steals `Image::data`
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        image.sampler = ImageSampler::Descriptor(if nearest {
            ImageSamplerDescriptor::nearest()
        } else {
            ImageSamplerDescriptor::linear()
        });
        Self {
            handle: images.add(image),
            w,
            h,
            rgba,
            half_res,
            // nothing is baked yet, so the first bake must cover everything
            force_full: true,
            // and it must happen on the first tick regardless of bake_hz
            accum: 1.0e9,
            sharp: Vec::new(),
            sig: None,
            cache: BakeCache::default(),
            ring: Vec::new(),
            mask: SlabMask::default(),
            last_dirty: Vec::new(),
            extra: Vec::new(),
            touched: Vec::new(),
            touch_set: Vec::new(),
            // the image was built from this exact buffer, so nothing is owed yet
            upload: Upload::default(),
            schematic: SchematicCache::default(),
        }
    }

    /// Consumes `layer.chunks.take_dirty()` and rewrites only those tiles.
    pub fn bake(
        &mut self,
        layer: &mut Layer,
        table: &MaterialTable,
        pal: &Palette,
        t: &LayerTreatment,
    ) {
        self.bake_with(layer, table, pal, t, false, false);
    }

    /// `bake` plus the two switches the plugin's config owns: the sleeping-cell debug
    /// tint and the x-ray outline pass. No slab depth — see [`LayerCanvas::bake_ctx`].
    pub fn bake_with(
        &mut self,
        layer: &mut Layer,
        table: &MaterialTable,
        pal: &Palette,
        t: &LayerTreatment,
        tint_sleeping: bool,
        outline: bool,
    ) {
        self.bake_ctx(
            layer,
            &BakeCtx {
                table,
                pal,
                treat: t,
                tint_sleeping,
                outline,
                extrude: None,
                cast: None,
                min_blur_px: 0.0,
            },
        );
    }

    /// The full bake: sharp pixels, then the slab lip, then anything cast onto this layer
    /// from the layer in front, then the depth-of-field blur.
    ///
    /// Pass order is not arbitrary. The lip has to exist before the cast, so that a slab's
    /// own face receives shadow like everything else; both have to happen before the blur,
    /// so the blur softens them along with the rest and no seam appears between a shadowed
    /// tile and its neighbour.
    pub fn bake_ctx(&mut self, layer: &mut Layer, ctx: &BakeCtx) {
        let BakeCtx {
            table,
            pal,
            treat: t,
            tint_sleeping,
            outline,
            ..
        } = *ctx;
        let (tint_sleeping, outline) = (tint_sleeping, outline);
        // Always drain, even for a hidden layer or a full rebake: the sim's dirty list has
        // exactly one reader, so anything left behind is lost, not deferred.
        let dirty = layer.chunks.take_dirty();
        if t.hidden {
            return;
        }
        debug_assert_eq!(self.w, layer.w, "canvas and layer must agree on width");
        debug_assert_eq!(self.h, layer.h, "canvas and layer must agree on height");

        let pt = PixelTreat::new(t);
        let step: u16 = if self.half_res { 2 } else { 1 };
        // masking the flag off here is why `cell_color` needs no config parameter
        let sleep_mask = if tint_sleeping { FLAG_SLEEPING } else { 0 };
        // A resampled plane needs its low-pass whatever the art asks for, so the geometric
        // floor from `depth::shimmer_safe_blur_px` wins ties against the treatment.
        let blur_px = t.blur_px.max(ctx.min_blur_px);
        let blur = blur_px >= BLUR_MIN_PX;
        let (tw, th) = (self.w, self.h);
        let area = tw as usize * th as usize;
        if area == 0 {
            return;
        }

        // A treatment change repaints every pixel, which no dirty list can express. So do
        // a change of lip depth and a change of cast strength, which is why both are in
        // the fingerprint.
        let sig = BakeSig::new(
            t,
            self.half_res,
            outline,
            tint_sleeping,
            ctx.extrude.map(|(_, d)| d).unwrap_or(0.0),
            ctx.cast.as_ref().map(|c| c.strength()).unwrap_or(0.0),
            blur,
        );
        if self.sig != Some(sig) {
            self.sig = Some(sig);
            self.cache.invalidate();
            self.force_full = true;
        }
        // the blur's shadow buffer has to exist, and be complete, before a tile blur can
        // read an apron out of it
        if blur && self.sharp.len() != self.rgba.len() {
            self.sharp = vec![0u8; self.rgba.len()];
            self.force_full = true;
        }

        let full = self.force_full;
        let cw = layer.chunks.cw as usize;
        let whole: Rect = (0, 0, tw - 1, th - 1);

        // Every rect this bake touches, in cell space: the sim's dirty tiles plus anything
        // a caster in front has invalidated (`mark_dirty_rect`). The lip reaches
        // `extrude_px` cells past a changed cell and the cast reaches its own offset and
        // softness, so each rect is grown before use — same reasoning as the blur apron.
        self.touched.clear();
        if full {
            self.touched.push(whole);
        } else {
            // Dedupe by chunk before baking anything. The sim's dirty tiles and the tiles a
            // caster in front invalidated overlap almost entirely in practice — the same
            // material is falling through both layers — so baking the two lists separately
            // paid for the same pixels twice.
            if self.touch_set.len() != layer.chunks.len() {
                self.touch_set = vec![false; layer.chunks.len()];
            }
            let mut mark = |c: usize, set: &mut Vec<bool>, out: &mut Vec<usize>| {
                if let Some(slot) = set.get_mut(c) {
                    if !*slot {
                        *slot = true;
                        out.push(c);
                    }
                }
            };
            let mut marked: Vec<usize> = Vec::new();
            for &c in &dirty {
                mark(c, &mut self.touch_set, &mut marked);
            }
            let cast_reach = ctx.cast_reach();
            for &r in &self.extra {
                let g = grow_by(r, tw, th, cast_reach);
                let (cx0, cy0) = (g.0 / CHUNK_PX, g.1 / CHUNK_PX);
                let (cx1, cy1) = (g.2 / CHUNK_PX, g.3 / CHUNK_PX);
                for cy in cy0..=cy1 {
                    for cx in cx0..=cx1 {
                        mark(cy as usize * cw + cx as usize, &mut self.touch_set, &mut marked);
                    }
                }
            }
            let lip_reach = ctx.lip_reach();
            for c in marked {
                self.touch_set[c] = false;
                if let Some(r) = chunk_rect(c, cw, tw, th) {
                    self.touched.push(grow_by(r, tw, th, lip_reach));
                }
            }
        }
        self.extra.clear();
        // what the sim changed, before growing — the caster's footprint for the layer behind
        self.last_dirty.clear();
        if full {
            self.last_dirty.push(whole);
        } else {
            for &c in &dirty {
                if let Some(r) = chunk_rect(c, cw, tw, th) {
                    self.last_dirty.push(r);
                }
            }
        }

        // ---- sharp pixels --------------------------------------------------------
        {
            let Self {
                rgba,
                sharp,
                cache,
                touched,
                ..
            } = self;
            let dst: &mut [u8] = if blur { sharp } else { rgba };
            let (grain, lut) = cache.split(table);
            for &r in touched.iter() {
                bake_rect(
                    dst, tw, r, layer, table, pal, &pt, grain, lut, step, sleep_mask, outline,
                );
            }
        }

        // ---- the slab lip --------------------------------------------------------
        if let Some((cfg, depth)) = ctx.extrude {
            let Self {
                rgba,
                sharp,
                touched,
                ..
            } = self;
            let dst: &mut [u8] = if blur { sharp } else { rgba };
            for &r in touched.iter() {
                extrude_rect(dst, tw, r, layer, &cfg, depth);
            }
        }

        // ---- what the slab in front throws onto this one -------------------------
        if let Some(cast) = &ctx.cast {
            let Self {
                rgba,
                sharp,
                touched,
                ..
            } = self;
            let dst: &mut [u8] = if blur { sharp } else { rgba };
            for &r in touched.iter() {
                cast_rect(dst, tw, r, cast);
            }
        }

        // ---- blur out of `sharp`, into the texture -------------------------------
        if blur {
            let Self {
                rgba,
                sharp,
                ring,
                touched,
                ..
            } = self;
            for &r in touched.iter() {
                // a changed cell also moves the blurred pixels one ring outside its own
                // rect, so the output rect is grown by the radius
                blur_rect(rgba, Some(sharp), tw, th, grow(r, tw, th), ring);
            }
        }

        // ---- what the GPU image now owes -----------------------------------------
        if full {
            self.upload.everything();
        } else {
            for i in 0..self.touched.len() {
                let r = self.touched[i];
                self.upload.add(if blur { grow(r, tw, th) } else { r }, area);
            }
        }

        self.force_full = false;
        self.accum = 0.0;
    }

    /// True when this canvas has pixels the `Image` has not been given yet.
    #[inline]
    pub fn needs_upload(&self) -> bool {
        self.upload.pending()
    }

    /// Bring this layer's coarse occupancy up to date, so the layer behind it can be cast
    /// onto. Must be called after the bake, whose `last_dirty` says what moved.
    pub fn update_mask(&mut self, layer: &Layer, softness: u8) {
        if self.mask.is_empty() {
            self.mask = SlabMask::for_layer(layer.w, layer.h);
            self.mask.update(layer, &[], true);
        } else if !self.last_dirty.is_empty() {
            let rects = std::mem::take(&mut self.last_dirty);
            self.mask.update(layer, &rects, false);
            self.last_dirty = rects;
        } else {
            return;
        }
        self.mask.soften(softness);
    }

    /// The rects the sim changed on the last bake — what the layer behind has to redo,
    /// because this layer's shadow moved with them.
    pub fn caster_rects(&self) -> &[Rect] {
        &self.last_dirty
    }

    /// This layer's coarse occupancy, for the layer behind it to be cast onto.
    pub fn mask(&self) -> &SlabMask {
        &self.mask
    }

    /// Invalidate a rect from outside: something that is not this layer's own cells changed
    /// what its pixels should look like. The next bake treats it exactly like a dirty tile.
    pub fn mark_dirty_rect(&mut self, r: Rect) {
        let r = (r.0.min(self.w - 1), r.1.min(self.h - 1), r.2.min(self.w - 1), r.3.min(self.h - 1));
        self.extra.push(r);
    }

    /// Zero the buffer. Used by the schematic canvas, which composites every layer into
    /// one texture and so has to start each bake empty. Also opens a new composite pass —
    /// see [`SchematicCache`].
    pub fn clear(&mut self) {
        self.rgba.fill(0);
        self.upload.everything();
        self.schematic.rewind();
    }

    /// The pixels, the coarse dimensions and the coarse cache as disjoint borrows, for
    /// [`crate::bake_schematic`]. It needs to composite into `rgba` while updating the
    /// cache, and `upload` is already owed everything by the `clear()` that must precede it.
    pub(crate) fn schematic_target(&mut self) -> (&mut [u8], u16, u16, &mut SchematicCache) {
        (&mut self.rgba, self.w, self.h, &mut self.schematic)
    }

    /// Copy the baked pixels into the `Image`, one dirty row-span at a time.
    ///
    /// Bevy 0.18 re-uploads a modified image whole, so this does not shrink the *GPU*
    /// transfer — but the CPU-side copy is 2 MB per 1024x512 layer per tick, which one
    /// changed 64x64 tile has no business paying.
    pub fn upload(&mut self, images: &mut Assets<Image>) {
        let Some(image) = images.get_mut(&self.handle) else {
            // the handle is gone; keep owing the pixels rather than dropping them
            return;
        };
        match image.data.as_mut() {
            Some(dst) if dst.len() == self.rgba.len() => {
                if self.upload.all {
                    dst.copy_from_slice(&self.rgba);
                } else {
                    let stride = self.w as usize * 4;
                    for &(x0, y0, x1, y1) in &self.upload.rects {
                        let a = x0 as usize * 4;
                        let b = (x1 as usize + 1) * 4;
                        for y in y0 as usize..=y1 as usize {
                            let row = y * stride;
                            dst[row + a..row + b].copy_from_slice(&self.rgba[row + a..row + b]);
                        }
                    }
                }
            }
            _ => image.data = Some(self.rgba.clone()),
        }
        self.upload.nothing();
    }
}

/// Inclusive cell rect of chunk `c`, clipped to the canvas; `None` if it falls outside.
#[inline]
fn chunk_rect(c: usize, cw: usize, tw: u16, th: u16) -> Option<Rect> {
    if cw == 0 {
        return None;
    }
    let x0 = (c % cw) as u16 * CHUNK_PX;
    let y0 = (c / cw) as u16 * CHUNK_PX;
    if x0 >= tw || y0 >= th {
        return None;
    }
    Some((
        x0,
        y0,
        (x0 + CHUNK_PX - 1).min(tw - 1),
        (y0 + CHUNK_PX - 1).min(th - 1),
    ))
}

/// Grow a rect by the blur radius (1px), clamped to the canvas.
#[inline]
fn grow_by(r: Rect, tw: u16, th: u16, n: u16) -> Rect {
    (
        r.0.saturating_sub(n),
        r.1.saturating_sub(n),
        r.2.saturating_add(n).min(tw - 1),
        r.3.saturating_add(n).min(th - 1),
    )
}

fn grow(r: Rect, tw: u16, th: u16) -> Rect {
    (
        r.0.saturating_sub(1),
        r.1.saturating_sub(1),
        (r.2 + 1).min(tw - 1),
        (r.3 + 1).min(th - 1),
    )
}

/// Per-canvas bake memos.
///
/// Both are pure functions of things that do not change from tick to tick, so they are
/// memoised rather than recomputed half a million times a frame.
#[derive(Default)]
struct BakeCache {
    /// `grain_of` per material id. The material table is fixed for the run — the same
    /// assumption `Palette::from_table` already makes.
    grain: Option<Box<[(f32, u8); 256]>>,
    lut: TreatLut,
}

impl BakeCache {
    fn invalidate(&mut self) {
        self.lut.invalidate();
    }

    /// The two memos as disjoint borrows, with the grain table built on first use.
    fn split(&mut self, table: &MaterialTable) -> (&[(f32, u8); 256], &mut TreatLut) {
        let grain = self.grain.get_or_insert_with(|| {
            let mut g = Box::new([(0.0f32, 0u8); 256]);
            for m in 0..256usize {
                g[m] = grain_of(table, m as u8);
            }
            g
        });
        (grain, &mut self.lut)
    }
}

/// `(material, grain jitter) -> fully treated RGB`, filled one material at a time.
///
/// Only sound for cells whose aux channels are all zero — which is the overwhelming
/// majority: dry, unstained, unworn, carrying nothing. For those, the pixel's colour is a
/// function of the material and the 256-valued grain jitter and nothing else, so a tile of
/// one material costs 256 `PixelTreat::apply` calls rather than 4096. Invalidated whenever
/// the treatment fingerprint moves, so it can never outlive the treatment it was built for.
#[derive(Default)]
struct TreatLut {
    /// 256 materials x 256 jitter values x RGB; allocated on first use
    rgb: Vec<u8>,
    /// which material rows are filled
    ready: [u64; 4],
}

impl TreatLut {
    fn invalidate(&mut self) {
        self.ready = [0; 4];
    }

    #[inline]
    fn get(&mut self, m: u8, jitter: u8, base: [u8; 3], amp: f32, pt: &PixelTreat) -> [u8; 3] {
        let mi = m as usize;
        if self.ready[mi >> 6] >> (mi & 63) & 1 == 0 {
            self.fill(mi, base, amp, pt);
        }
        let o = (mi * 256 + jitter as usize) * 3;
        [self.rgb[o], self.rgb[o + 1], self.rgb[o + 2]]
    }

    #[cold]
    fn fill(&mut self, mi: usize, base: [u8; 3], amp: f32, pt: &PixelTreat) {
        if self.rgb.len() != 256 * 256 * 3 {
            self.rgb = vec![0u8; 256 * 256 * 3];
        }
        for j in 0..256usize {
            let rgb = pt.apply(speckle_byte(base, amp, j as u8));
            let o = (mi * 256 + j) * 3;
            self.rgb[o..o + 3].copy_from_slice(&rgb);
        }
        self.ready[mi >> 6] |= 1u64 << (mi & 63);
    }
}

/// Wetness a submerged powder is drawn at before it has absorbed anything of its own — a
/// grain underwater reads as soaked because it is, even though the conserved wetness
/// channel is still zero. Close to the ~240 an absorbed bed reaches, so a sinking grain and
/// the deposit that later lands on it look the same rather than popping from dry to wet.
const SUBMERGED_WET: u8 = 210;

/// `true` when the powder at `(x, y)` has a liquid cell against one of its four faces, i.e.
/// it is sitting in water. A display cue only — absorption still owns the wetness channel.
#[inline]
fn submerged_powder(l: &Layer, table: &MaterialTable, m: u8, x: u16, y: u16) -> bool {
    if table.class(m) != MaterialClass::Powder {
        return false;
    }
    for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
        if l.in_bounds(nx, ny)
            && table.class(l.mat_at(nx as u16, ny as u16)) == MaterialClass::Liquid
        {
            return true;
        }
    }
    false
}

/// Bake one inclusive cell rect into `rgba`.
///
/// `step` is the `half_res` stride: one cell is sampled per `step`x`step` block and the
/// whole block is written from it.
#[allow(clippy::too_many_arguments)]
fn bake_rect(
    rgba: &mut [u8],
    tw: u16,
    rect: Rect,
    layer: &Layer,
    table: &MaterialTable,
    pal: &Palette,
    pt: &PixelTreat,
    grain: &[(f32, u8); 256],
    lut: &mut TreatLut,
    step: u16,
    sleep_mask: u8,
    outline: bool,
) {
    let (x0, y0, x1, y1) = rect;
    let stride = tw as usize * 4;
    let lw = layer.w as usize;
    let mut y = y0;
    while y <= y1 {
        let base = y as usize * lw;
        let mut x = x0;
        while x <= x1 {
            let i = base + x as usize;
            let m = layer.mat[i];

            let (rgb, a) = if m == 0 {
                // air is transparent, not black: the cutaway only reads if the layers
                // behind show through the holes you dug
                ([0u8, 0, 0], 0u8)
            } else {
                let (amp, shift) = grain[m as usize];
                // A powder sitting in water reads as wet even before it has absorbed a
                // thing: it is underwater. Absorption still owns the wetness *channel* and
                // the conserved water budget — this only lifts the drawn value so a grain
                // does not look bone dry while submerged (a lone grain never reaches the
                // whole-cell capacity absorption needs, so its channel stays 0 for good).
                let wet = if layer.wetness[i] < SUBMERGED_WET
                    && submerged_powder(layer, table, m, x, y)
                {
                    SUBMERGED_WET
                } else {
                    layer.wetness[i]
                };
                // Aux channels are zero for almost every cell, and when they are, the
                // whole per-pixel chain collapses to a table lookup. `susp_mat` only
                // speaks through `susp_conc`; `charred` and `head` are not read at all.
                let aux = wet
                    | layer.dirt[i]
                    | layer.wear[i]
                    | layer.susp_conc[i]
                    | (layer.flags[i] & sleep_mask);
                let rgb = if aux == 0 {
                    let jitter = if amp > 0.0 {
                        grain_jitter(m, shift, x, y)
                    } else {
                        0
                    };
                    lut.get(m, jitter, pal.rgb[m as usize], amp, pt)
                } else {
                    let rgb = pal.cell_color(
                        m,
                        layer.flags[i] & sleep_mask,
                        wet,
                        layer.dirt[i],
                        layer.wear[i],
                        layer.susp_mat[i],
                        layer.susp_conc[i],
                    );
                    pt.apply(speckle_byte(rgb, amp, grain_jitter(m, shift, x, y)))
                };

                if outline {
                    // ghost the fill, keep the silhouette: the "colour-coded outlines"
                    // half of the vault's x-ray rule. The colour itself comes from the
                    // per-depth hue shift already folded into `pt`.
                    if is_edge(layer, x, y) {
                        (
                            [
                                (rgb[0] as f32 * OUTLINE_BOOST).min(255.0) as u8,
                                (rgb[1] as f32 * OUTLINE_BOOST).min(255.0) as u8,
                                (rgb[2] as f32 * OUTLINE_BOOST).min(255.0) as u8,
                            ],
                            255,
                        )
                    } else {
                        (rgb, XRAY_FILL_ALPHA)
                    }
                } else {
                    (rgb, 255)
                }
            };

            let px = [rgb[0], rgb[1], rgb[2], a];
            if step == 1 {
                let o = y as usize * stride + x as usize * 4;
                rgba[o..o + 4].copy_from_slice(&px);
            } else {
                // replicate across the step x step block
                let bx1 = (x + step - 1).min(x1);
                let by1 = (y + step - 1).min(y1);
                for by in y..=by1 {
                    let row = by as usize * stride;
                    for bx in x..=bx1 {
                        let o = row + bx as usize * 4;
                        rgba[o..o + 4].copy_from_slice(&px);
                    }
                }
            }

            x += step;
        }
        y += step;
    }
}

/// Draw the slab lip over one rect: air pixels that sit in the shadow-side lip of a solid
/// cell take that cell's colour, darkened.
///
/// Reads occupancy from the layer and colour from pixels this bake has already written, so
/// it must run after [`bake_rect`] over the same rect. It writes only where the cell is
/// air, so it can never overwrite a real cell — the lip lives in the space the cutaway was
/// showing through, which is exactly where a real slab's side face would be.
fn extrude_rect(rgba: &mut [u8], tw: u16, rect: Rect, layer: &Layer, cfg: &ExtrudeConfig, depth: f32) {
    if depth < 1.0 {
        return;
    }
    let (x0, y0, x1, y1) = rect;
    let stride = tw as usize * 4;
    let lw = layer.w as usize;
    for y in y0..=y1 {
        let base = y as usize * lw;
        for x in x0..=x1 {
            if layer.mat[base + x as usize] != 0 {
                continue;
            }
            let Some((sx, sy, shade)) = lip_at(layer, cfg, x, y, depth) else {
                continue;
            };
            let src = sy as usize * stride + sx as usize * 4;
            // the source may itself be air-adjacent lip from a previous rect; its alpha is
            // what tells us whether it holds a real colour yet
            if rgba[src + 3] == 0 {
                continue;
            }
            let k = (1.0 - shade).clamp(0.0, 1.0);
            let px = [
                (rgba[src] as f32 * k) as u8,
                (rgba[src + 1] as f32 * k) as u8,
                (rgba[src + 2] as f32 * k) as u8,
                255,
            ];
            let o = y as usize * stride + x as usize * 4;
            rgba[o..o + 4].copy_from_slice(&px);
        }
    }
}

/// Multiply one rect by the shadow the slab in front throws onto it.
///
/// Alpha is untouched: a shadow darkens what is there, it does not paint on air. That is
/// what keeps a cast from filling the holes the cutaway needs — an unexcavated gap between
/// two slabs stays a gap, it just gets darker where something is above it.
fn cast_rect(rgba: &mut [u8], tw: u16, rect: Rect, cast: &Cast) {
    if cast.mask.is_empty() || cast.cfg.strength <= 0.0 {
        return;
    }
    let (x0, y0, x1, y1) = rect;
    let stride = tw as usize * 4;
    let inv = 1.0 / MASK_DIV as f32;
    let (ox, oy) = cast.offset;
    // One shadow value per 2x2 pixel block. The mask is already coarse (one cell per
    // `MASK_DIV` px) and softened, so its output has nothing finer than a few px in it —
    // sampling it per pixel was paying four bilinear fetches to reproduce a value that
    // barely moves. Blocking it quarters the cast's cost for no visible difference; going
    // further, to one sample per mask cell, does start to show as stair-steps.
    let mut y = y0;
    while y <= y1 {
        let my = (y as f32 + 0.5 - oy) * inv;
        let cy = (y as f32 + 0.5) * inv;
        let mut x = x0;
        while x <= x1 {
            let occ = cast.mask.sample((x as f32 + 0.5 - ox) * inv, my);
            // Contact shading: material *directly* in front, with no displacement, is the
            // caster touching this surface rather than hanging over it. Without this the
            // stack reads as two lit images with a smudge between them.
            let contact = cast.mask.sample((x as f32 + 0.5) * inv, cy);
            let dark = (occ * cast.cfg.strength + contact * cast.cfg.contact).clamp(0.0, 0.92);
            if dark > 0.002 {
                let k = 1.0 - dark;
                // Shadows are cool, not black. A straight multiply toward black reads as
                // grime on a mid-grey material — which is most of a vessel cut out of rock —
                // so the darkened pixel also leans slightly toward the fog colour. Same
                // reasoning as `treat::FOG`: shadow is air you are seeing *through*.
                let tint = dark * SHADOW_TINT;
                let by1 = (y + 1).min(y1);
                let bx1 = (x + 1).min(x1);
                for by in y..=by1 {
                    let row = by as usize * stride;
                    for bx in x..=bx1 {
                        let o = row + bx as usize * 4;
                        if rgba[o + 3] == 0 {
                            continue;
                        }
                        let (r, g, b) = (rgba[o] as f32 * k, rgba[o + 1] as f32 * k, rgba[o + 2] as f32 * k);
                        rgba[o] = (r + (crate::treat::FOG[0] * 255.0 - r) * tint) as u8;
                        rgba[o + 1] = (g + (crate::treat::FOG[1] * 255.0 - g) * tint) as u8;
                        rgba[o + 2] = (b + (crate::treat::FOG[2] * 255.0 - b) * tint) as u8;
                    }
                }
            }
            x += 2;
        }
        y += 2;
    }
}

/// Does this cell border empty space (or the edge of the vessel)? Drives the x-ray
/// silhouette.
#[inline]
fn is_edge(layer: &Layer, x: u16, y: u16) -> bool {
    const N: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for (dx, dy) in N {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;
        if !layer.in_bounds(nx, ny) {
            return true;
        }
        if layer.mat[layer.idx(nx as u16, ny as u16)] == 0 {
            return true;
        }
    }
    false
}

/// 3x3 CPU box blur over an RGBA8 buffer, used for treatments with `blur_px >= 2`.
///
/// Blurs in premultiplied alpha and unpremultiplies on the way out. Blurring straight
/// RGBA would pull the transparent black of air into every silhouette edge and ring the
/// layer with dark fringes — which on a dim rear layer is the difference between "soft"
/// and "grubby".
///
/// Separable: a horizontal 3-tap pass into a three-row ring, then a vertical 3-tap pass
/// out of it. Three rows is exactly how far back the vertical pass ever looks, which is
/// why this needs no copy of the image and can run in place. It is *bit-exact* against the
/// naive 9-tap window rather than merely close: every premultiplied term is below 2^24 and
/// at most nine are summed, so splitting the window into row sums cannot round differently.
pub fn box_blur_rgba(rgba: &mut [u8], w: u16, h: u16) {
    if w == 0 || h == 0 {
        return;
    }
    let mut ring = Vec::new();
    blur_rect(rgba, None, w, h, (0, 0, w - 1, h - 1), &mut ring);
}

/// Blur `rect` of `src` — or of `dst` itself, when `src` is `None` — into `dst`.
///
/// `ring` is the caller's scratch, so a per-tile blur allocates nothing at all.
fn blur_rect(dst: &mut [u8], src: Option<&[u8]>, w: u16, h: u16, rect: Rect, ring: &mut Vec<u32>) {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 {
        return;
    }
    let (x0, y0) = (rect.0 as usize, rect.1 as usize);
    let (x1, y1) = ((rect.2 as usize).min(w - 1), (rect.3 as usize).min(h - 1));
    if x0 > x1 || y0 > y1 {
        return;
    }
    let span = x1 - x0 + 1;
    // three rows of horizontal sums, plus one row of premultiplied pixels feeding them
    let need = span * 4 * 3 + (span + 2) * 4;
    if ring.len() < need {
        ring.resize(need, 0);
    }
    let (rows, pre) = ring.split_at_mut(span * 4 * 3);

    // highest source row whose horizontal sums are in the ring
    let mut done: isize = -1;
    for y in y0..=y1 {
        let ya = y.saturating_sub(1);
        let yb = (y + 1).min(h - 1);
        if done < ya as isize {
            done = ya as isize - 1;
        }
        while done < yb as isize {
            done += 1;
            let r = done as usize;
            let slot = (r % 3) * span * 4;
            let out = &mut rows[slot..slot + span * 4];
            match src {
                // `&*dst` is a shared reborrow that ends with the call, which is what
                // makes the in-place case sound: the vertical pass writes row `y` while
                // the horizontal pass has only ever read rows up to `y + 1`, and row
                // `y + 1` is summed before it is written.
                None => hsum_row(out, pre, &*dst, w, r, x0, x1),
                Some(s) => hsum_row(out, pre, s, w, r, x0, x1),
            }
        }

        let ny = (yb - ya + 1) as u32;
        // this row's ring slots, hoisted — the modulo has no business being inside the
        // per-pixel loop, and neither does the variable trip count
        let (sa, sc, sb) = ((ya % 3) * span * 4, (y % 3) * span * 4, (yb % 3) * span * 4);
        for x in x0..=x1 {
            let nx = ((x + 1).min(w - 1) - x.saturating_sub(1) + 1) as u32;
            let n = (nx * ny) as f32;
            let k = (x - x0) * 4;
            let mut acc = [0u32; 4];
            for j in 0..4 {
                acc[j] = rows[sc + k + j];
            }
            if ya != y {
                for j in 0..4 {
                    acc[j] += rows[sa + k + j];
                }
            }
            if yb != y {
                for j in 0..4 {
                    acc[j] += rows[sb + k + j];
                }
            }
            let o = (y * w + x) * 4;
            if acc[3] > 0 {
                let ia = acc[3] as f32;
                dst[o] = (acc[0] as f32 / ia).clamp(0.0, 255.0) as u8;
                dst[o + 1] = (acc[1] as f32 / ia).clamp(0.0, 255.0) as u8;
                dst[o + 2] = (acc[2] as f32 / ia).clamp(0.0, 255.0) as u8;
            } else {
                dst[o] = 0;
                dst[o + 1] = 0;
                dst[o + 2] = 0;
            }
            dst[o + 3] = (acc[3] as f32 / n).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Premultiplied horizontal 3-tap sums for one row: `(r*a, g*a, b*a, a)` summed over the
/// clamped window `[x-1, x+1]`, for each `x` in `x0..=x1`.
///
/// Two stages, because one fused stage keeps the multiplies and the neighbour-gathering in
/// the same loop and neither vectorises. Stage one premultiplies the window `x0-1 ..= x1+1`
/// into `pre`; stage two is a pure elementwise 3-tap over contiguous `u32`s, which the
/// autovectoriser does take. Both are exact integer arithmetic — every term is at most
/// `255*255` and at most nine are ever summed, well inside `u32` and inside the 24 bits an
/// `f32` represents exactly, which is what makes the separable blur bit-identical to a
/// 9-tap window.
#[inline]
fn hsum_row(
    out: &mut [u32],
    pre: &mut [u32],
    src: &[u8],
    w: usize,
    y: usize,
    x0: usize,
    x1: usize,
) {
    let row = y * w * 4;
    // `pre` runs from x0-1 to x1+1, clamped at the image edge, so index 0 is x0-1
    let lo = x0.saturating_sub(1);
    let hi = (x1 + 1).min(w - 1);
    for (k, sx) in (lo..=hi).enumerate() {
        let o = row + sx * 4;
        let a = src[o + 3] as u32;
        let p = &mut pre[k * 4..k * 4 + 4];
        p[0] = src[o] as u32 * a;
        p[1] = src[o + 1] as u32 * a;
        p[2] = src[o + 2] as u32 * a;
        p[3] = a;
    }

    // the clamped window is a full three taps everywhere except hard against the image
    // edge, and only that interior run is worth vectorising
    let mid0 = x0.max(1);
    let mid1 = if w >= 2 { x1.min(w - 2) } else { 0 };
    if mid0 <= mid1 {
        for x in mid0..=mid1 {
            let c = (x - lo) * 4;
            let k = (x - x0) * 4;
            for j in 0..4 {
                out[k + j] = pre[c - 4 + j] + pre[c + j] + pre[c + 4 + j];
            }
        }
    }
    if x0 == 0 {
        // window {0, 1}, or just {0} on a 1px-wide image
        let b = 1usize.min(w - 1) * 4;
        for j in 0..4 {
            out[j] = pre[j] + if b != 0 { pre[b + j] } else { 0 };
        }
    }
    if x1 == w - 1 && w >= 2 {
        // window {w-2, w-1}
        let c = (x1 - lo) * 4;
        let k = (x1 - x0) * 4;
        for j in 0..4 {
            out[k + j] = pre[c - 4 + j] + pre[c + j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 9-tap window the separable blur replaced, kept verbatim as the reference the
    /// fast path is held to.
    fn naive_blur(rgba: &mut [u8], w: u16, h: u16) {
        let (w, h) = (w as usize, h as usize);
        if w == 0 || h == 0 {
            return;
        }
        let src = rgba.to_vec();
        let at = |x: usize, y: usize| (y * w + x) * 4;
        for y in 0..h {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(h - 1);
            for x in 0..w {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(w - 1);
                let mut acc = [0.0f32; 4];
                let mut n = 0.0f32;
                for sy in y0..=y1 {
                    for sx in x0..=x1 {
                        let o = at(sx, sy);
                        let a = src[o + 3] as f32;
                        acc[0] += src[o] as f32 * a;
                        acc[1] += src[o + 1] as f32 * a;
                        acc[2] += src[o + 2] as f32 * a;
                        acc[3] += a;
                        n += 1.0;
                    }
                }
                let o = at(x, y);
                if acc[3] > 0.0 {
                    rgba[o] = (acc[0] / acc[3]).clamp(0.0, 255.0) as u8;
                    rgba[o + 1] = (acc[1] / acc[3]).clamp(0.0, 255.0) as u8;
                    rgba[o + 2] = (acc[2] / acc[3]).clamp(0.0, 255.0) as u8;
                } else {
                    rgba[o] = 0;
                    rgba[o + 1] = 0;
                    rgba[o + 2] = 0;
                }
                rgba[o + 3] = (acc[3] / n).clamp(0.0, 255.0) as u8;
            }
        }
    }

    fn noise(w: u16, h: u16, seed: u64) -> Vec<u8> {
        let mut s = seed | 1;
        (0..w as usize * h as usize * 4)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let v = (s >> 24) as u8;
                // plenty of fully transparent pixels, which is where the premultiplied
                // fallback branch lives
                if v < 40 { 0 } else { v }
            })
            .collect()
    }

    /// The separable blur is an optimisation, not a look change: bit-identical to the
    /// 9-tap window it replaced, on awkward sizes and across transparent runs.
    #[test]
    fn blur_matches_the_naive_window() {
        for (w, h) in [
            (1u16, 1u16),
            (2, 1),
            (1, 5),
            (3, 3),
            (7, 4),
            (16, 9),
            (64, 64),
        ] {
            let base = noise(w, h, w as u64 * 7919 + h as u64);
            let mut a = base.clone();
            let mut b = base;
            naive_blur(&mut a, w, h);
            box_blur_rgba(&mut b, w, h);
            assert_eq!(a, b, "separable blur diverged at {w}x{h}");
        }
    }

    /// Blurring sub-rects out of a shadow buffer must reproduce a whole-image blur exactly
    /// — that is the claim the tile blur rests on, and the reason `sharp` exists.
    #[test]
    fn tile_blur_matches_a_whole_image_blur() {
        let (w, h) = (48u16, 32u16);
        let src = noise(w, h, 12345);
        let mut whole = src.clone();
        box_blur_rgba(&mut whole, w, h);

        let mut tiled = vec![0u8; src.len()];
        let mut ring = Vec::new();
        // ragged rects, including ones against every border
        // a ragged partition of the image, including rects against every border
        for r in [
            (0, 0, 15, 15),
            (16, 0, 47, 9),
            (16, 10, 20, 15),
            (21, 10, 47, 31),
            (0, 16, 20, 31),
        ] {
            blur_rect(&mut tiled, Some(&src), w, h, r, &mut ring);
        }
        assert_eq!(whole, tiled, "tile blur seamed against a whole-image blur");
    }

    /// A scene with every path in `bake_rect` represented: air, several materials, and
    /// cells carrying each aux channel, so the `(material, jitter)` memo and the
    /// full `cell_color` fallback are both exercised.
    fn rich_layer(t: &MaterialTable) -> Layer {
        let sand = t.id("sand").unwrap().0;
        let water = t.id("water").unwrap().0;
        let (w, h) = (150u16, 90u16);
        let mut l = Layer::new(w, h, pixelsim::LayerSlot::Plant, 3);
        let mut s = 0x243f_6a88_85a3_08d3u64;
        for y in 0..h {
            for x in 0..w {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let i = l.idx(x, y);
                l.mat[i] = match s % 5 {
                    0 | 1 => sand,
                    2 => water,
                    3 => 0,
                    _ => sand,
                };
                // sparse aux, so most cells take the memo and some do not
                if s % 11 == 0 {
                    l.wetness[i] = (s >> 8) as u8;
                }
                if s % 13 == 0 {
                    l.dirt[i] = (s >> 16) as u8;
                }
                if s % 17 == 0 {
                    l.wear[i] = (s >> 24) as u8;
                }
                if s % 19 == 0 {
                    l.susp_mat[i] = water;
                    l.susp_conc[i] = (s >> 32) as u8;
                }
                if s % 23 == 0 {
                    l.flags[i] |= FLAG_SLEEPING;
                }
            }
        }
        l
    }

    /// The whole point of the rewrite: baking tile by tile must land on exactly the pixels
    /// a whole-layer bake would have produced — for the blurred treatments too, which is
    /// where the shadow buffer and the 1px apron earn their keep. Anything else is a visual
    /// change wearing an optimisation's clothes.
    #[test]
    fn tile_bakes_reproduce_a_full_bake() {
        let t = MaterialTable::embedded();
        let pal = Palette::from_table(&t);
        let mut images = Assets::<Image>::default();
        let cfg = crate::treat::RenderConfig::default();

        for (name, tr, outline, tint) in [
            ("depth 0", LayerTreatment::at_depth(0), false, false),
            ("depth 0 active", LayerTreatment::at_depth(0).emphasized(1.0, LayerTreatment::at_depth(0)), false, false),
            ("depth 2 blurred", LayerTreatment::at_depth(2), false, false),
            ("depth 5 blurred half-res", LayerTreatment::at_depth(5), false, false),
            (
                "xray",
                cfg.treatment_for(1, 1, pixelsim::LayerSlot::Plant, true),
                true,
                false,
            ),
            ("sleeping tint", LayerTreatment::at_depth(0), false, true),
        ] {
            let mut full_layer = rich_layer(&t);
            let mut tiled_layer = rich_layer(&t);
            let (w, h) = (full_layer.w, full_layer.h);

            let mut full = LayerCanvas::new(&mut images, w, h, tr.half_res, true);
            full.bake_with(&mut full_layer, &t, &pal, &tr, tint, outline);

            // same treatment, but reached one dirty chunk at a time after a cold full bake
            let mut tiled = LayerCanvas::new(&mut images, w, h, tr.half_res, true);
            tiled.bake_with(&mut tiled_layer, &t, &pal, &tr, tint, outline);
            for c in 0..tiled_layer.chunks.len() {
                let r = tiled_layer.chunks.bounds(c, w, h);
                // change something real inside the chunk, then rebake only that chunk
                let i = tiled_layer.idx(r.x0 + 1, r.y0 + 1);
                tiled_layer.mat[i] = t.id("water").unwrap().0;
                tiled_layer.wetness[i] = 200;
                let j = full_layer.idx(r.x0 + 1, r.y0 + 1);
                full_layer.mat[j] = t.id("water").unwrap().0;
                full_layer.wetness[j] = 200;

                tiled_layer.chunks.mark_dirty(r.x0 + 1, r.y0 + 1);
                tiled.bake_with(&mut tiled_layer, &t, &pal, &tr, tint, outline);
            }
            full.force_full = true;
            full.bake_with(&mut full_layer, &t, &pal, &tr, tint, outline);

            let diff = full
                .rgba
                .chunks(4)
                .zip(tiled.rgba.chunks(4))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(diff, 0, "{name}: {diff} pixels differ from a full bake");
        }
    }

    /// The memo is only allowed to be a memo: a canvas that has just been asked for a
    /// different treatment must not serve colours built for the old one.
    #[test]
    fn a_treatment_change_repaints_everything() {
        let t = MaterialTable::embedded();
        let pal = Palette::from_table(&t);
        let mut images = Assets::<Image>::default();
        let mut layer = rich_layer(&t);
        let (w, h) = (layer.w, layer.h);

        let mut a = LayerCanvas::new(&mut images, w, h, false, true);
        a.bake_with(
            &mut layer,
            &t,
            &pal,
            &LayerTreatment::at_depth(0),
            false,
            false,
        );
        // nothing is dirty from here on; only the treatment moves
        a.bake_with(
            &mut layer,
            &t,
            &pal,
            &LayerTreatment::at_depth(2),
            false,
            false,
        );

        let mut b = LayerCanvas::new(&mut images, w, h, false, true);
        b.bake_with(
            &mut layer,
            &t,
            &pal,
            &LayerTreatment::at_depth(2),
            false,
            false,
        );

        assert_eq!(a.rgba, b.rgba, "a treatment change left stale pixels");
    }

    #[test]
    fn blur_leaves_a_uniform_opaque_field_alone() {
        let (w, h) = (8u16, 8u16);
        let mut rgba = vec![0u8; w as usize * h as usize * 4];
        for p in rgba.chunks_mut(4) {
            p.copy_from_slice(&[120, 60, 30, 255]);
        }
        let before = rgba.clone();
        box_blur_rgba(&mut rgba, w, h);
        assert_eq!(rgba, before, "a flat field must survive the blur unchanged");
    }

    /// Blurring straight RGBA would drag the transparent black of air into every edge.
    /// Premultiplied blur must leave the *colour* of an isolated opaque pixel intact and
    /// only spread its alpha.
    #[test]
    fn blur_does_not_darken_edges_against_transparent_air() {
        let (w, h) = (3u16, 3u16);
        let mut rgba = vec![0u8; w as usize * h as usize * 4];
        let centre = (1 * 3 + 1) * 4;
        rgba[centre..centre + 4].copy_from_slice(&[200, 100, 50, 255]);

        box_blur_rgba(&mut rgba, w, h);

        // colour preserved everywhere it is visible at all
        for p in rgba.chunks(4) {
            if p[3] > 0 {
                assert_eq!(&p[0..3], &[200, 100, 50], "colour bled toward black");
            }
        }
        // alpha spread to the neighbours
        assert!(rgba[3] > 0 && rgba[3] < 255, "alpha should have feathered");
    }

    #[test]
    fn blur_handles_degenerate_sizes() {
        let mut empty: Vec<u8> = Vec::new();
        box_blur_rgba(&mut empty, 0, 0);
        let mut one = vec![9u8, 8, 7, 255];
        box_blur_rgba(&mut one, 1, 1);
        assert_eq!(one, vec![9, 8, 7, 255]);
    }
}

#[cfg(test)]
mod slab_pass_tests {
    use super::*;
    use crate::palette::Palette;
    use crate::slab::MASK_DIV;
    use crate::treat::{LayerTreatment, RenderConfig};
    use bevy::asset::Assets;
    use bevy::image::Image;
    use pixelsim::{Layer, LayerSlot, MaterialTable};

    /// A solid block in the middle of an otherwise empty layer.
    fn block(t: &MaterialTable, slot: LayerSlot) -> Layer {
        let stone = t.id("stone").unwrap().0;
        let mut l = Layer::new(128, 128, slot, 11);
        for y in 40..80u16 {
            for x in 40..80u16 {
                let i = l.idx(x, y);
                l.mat[i] = stone;
            }
        }
        l
    }

    fn ctx<'a>(
        table: &'a MaterialTable,
        pal: &'a Palette,
        treat: &'a LayerTreatment,
        extrude: Option<(ExtrudeConfig, f32)>,
        cast: Option<Cast<'a>>,
    ) -> BakeCtx<'a> {
        BakeCtx {
            table,
            pal,
            treat,
            tint_sleeping: false,
            outline: false,
            extrude,
            cast,
            min_blur_px: 0.0,
        }
    }

    /// The lip has to appear in the air *outside* the block, on the shadow side only, and
    /// it must never overwrite a real cell — the whole point is that it occupies space the
    /// cutaway was seeing through.
    #[test]
    fn the_lip_is_drawn_into_air_on_the_shadow_side() {
        let t = MaterialTable::embedded();
        let pal = Palette::from_table(&t);
        let treat = LayerTreatment::at_depth(0);
        let cfg = ExtrudeConfig::default();
        let mut images = Assets::<Image>::default();
        let mut layer = block(&t, LayerSlot::Plant);

        let mut plain = LayerCanvas::new(&mut images, 128, 128, false, true);
        plain.bake_ctx(&mut layer, &ctx(&t, &pal, &treat, None, None));
        let without = plain.rgba.clone();

        let mut lipped = LayerCanvas::new(&mut images, 128, 128, false, true);
        lipped.bake_ctx(&mut layer, &ctx(&t, &pal, &treat, Some((cfg, 3.0)), None));
        let with = &lipped.rgba;

        let px = |buf: &[u8], x: usize, y: usize| {
            let o = (y * 128 + x) * 4;
            [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]
        };

        // below-right of the block: air before, opaque lip after
        assert_eq!(px(&without, 60, 81)[3], 0, "should have been air");
        assert_eq!(px(with, 60, 81)[3], 255, "lip did not appear below the block");
        // above-left: the light comes from there, so no lip
        assert_eq!(px(with, 60, 38)[3], 0, "lip appeared on the lit side");
        // inside the block: untouched
        assert_eq!(px(&without, 60, 60), px(with, 60, 60), "lip overwrote a real cell");
        // and it is darker than the face it came from
        let face = px(with, 60, 79);
        let lip = px(with, 60, 81);
        assert!(
            lip[0] < face[0] && lip[1] < face[1],
            "lip {lip:?} is not darker than its face {face:?}"
        );
    }

    /// A slab in front must darken the slab behind it, without painting on air — a cast
    /// that filled holes would close the cutaway the whole game is looking through.
    #[test]
    fn a_caster_darkens_the_layer_behind_without_filling_air() {
        let t = MaterialTable::embedded();
        let pal = Palette::from_table(&t);
        let treat = LayerTreatment::at_depth(1);
        // The *mechanism* under test, so the config is explicit: the shipped default turns
        // the baked cast off, because real geometry casts real shadows now. Reading the
        // default here would test the default rather than the pass.
        let shadow = ShadowConfig {
            strength: 0.55,
            offset_px: 6.0,
            softness: 2,
            contact: 0.25,
        };
        let cfg = RenderConfig {
            shadow,
            ..RenderConfig::default()
        };
        let mut images = Assets::<Image>::default();

        // front layer: the block. back layer: a full sheet, with one hole punched in it.
        let mut front = block(&t, LayerSlot::Plant);
        let stone = t.id("stone").unwrap().0;
        let mut back = Layer::new(128, 128, LayerSlot::Plant, 12);
        back.fill(stone);
        for y in 100..110u16 {
            for x in 50..60u16 {
                let i = back.idx(x, y);
                back.mat[i] = 0;
            }
        }

        let mut front_canvas = LayerCanvas::new(&mut images, 128, 128, false, true);
        front_canvas.bake_ctx(&mut front, &ctx(&t, &pal, &treat, None, None));
        front_canvas.update_mask(&front, cfg.shadow.softness);

        let mut lit = LayerCanvas::new(&mut images, 128, 128, false, true);
        lit.bake_ctx(&mut back, &ctx(&t, &pal, &treat, None, None));
        let unshadowed = lit.rgba.clone();

        let mut shadowed = LayerCanvas::new(&mut images, 128, 128, false, true);
        shadowed.bake_ctx(
            &mut back,
            &ctx(
                &t,
                &pal,
                &treat,
                None,
                Some(Cast {
                    mask: front_canvas.mask(),
                    cfg: cfg.shadow,
                    offset: (4.0, 6.0),
                }),
            ),
        );

        let at = |buf: &[u8], x: usize, y: usize| {
            let o = (y * 128 + x) * 4;
            (buf[o] as u32 + buf[o + 1] as u32 + buf[o + 2] as u32, buf[o + 3])
        };

        // under the caster: darker
        let (lit_v, _) = at(&unshadowed, 60, 60);
        let (dark_v, dark_a) = at(&shadowed.rgba, 60, 60);
        assert!(dark_v < lit_v, "cast did not darken: {lit_v} -> {dark_v}");
        assert_eq!(dark_a, 255, "cast must not change opacity");

        // far from the caster: untouched
        assert_eq!(at(&unshadowed, 5, 120), at(&shadowed.rgba, 5, 120));

        // the hole in the back sheet stays a hole even though the caster is over it
        assert_eq!(at(&shadowed.rgba, 55, 105).1, 0, "cast filled air");
    }

    /// A cue toggled off must leave the pixels exactly as a canvas that never had it. This
    /// is what makes the spike's F2/F3 A/B honest rather than a slow accumulation of
    /// shadow on shadow.
    #[test]
    fn the_slab_cues_do_not_accumulate_across_bakes() {
        let t = MaterialTable::embedded();
        let pal = Palette::from_table(&t);
        let treat = LayerTreatment::at_depth(0);
        let cfg = ExtrudeConfig::default();
        let mut images = Assets::<Image>::default();
        let mut layer = block(&t, LayerSlot::Plant);

        let mut canvas = LayerCanvas::new(&mut images, 128, 128, false, true);
        // lip on, then off again, twice
        for _ in 0..2 {
            canvas.bake_ctx(&mut layer, &ctx(&t, &pal, &treat, Some((cfg, 3.0)), None));
            canvas.bake_ctx(&mut layer, &ctx(&t, &pal, &treat, None, None));
        }
        let toggled = canvas.rgba.clone();

        let mut fresh = LayerCanvas::new(&mut images, 128, 128, false, true);
        fresh.bake_ctx(&mut layer, &ctx(&t, &pal, &treat, None, None));

        let diff = toggled
            .chunks(4)
            .zip(fresh.rgba.chunks(4))
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(diff, 0, "{diff} pixels differ after toggling the lip off");
    }

    /// The baked cues must ship *off*: slabs have real thickness and the sun casts real
    /// shadows through them, so a painted lip is a second disagreeing edge and a baked cast
    /// double-darkens whatever the real shadow already darkened.
    #[test]
    fn the_baked_cues_are_off_by_default() {
        let cfg = RenderConfig::default();
        assert_eq!(cfg.extrude.px, 0.0, "the painted lip is back on");
        assert_eq!(cfg.shadow.strength, 0.0, "the baked cast is back on");
    }

    /// The mask is the caster's occupancy, so it has to be current before the layer behind
    /// bakes — and after a cell moves, the layer behind has to be told, or its shadow is
    /// stale in exactly the tiles that changed.
    #[test]
    fn a_moved_cell_invalidates_the_shadow_behind_it() {
        let t = MaterialTable::embedded();
        let mut images = Assets::<Image>::default();
        let pal = Palette::from_table(&t);
        let treat = LayerTreatment::at_depth(0);
        let mut front = block(&t, LayerSlot::Plant);

        let mut canvas = LayerCanvas::new(&mut images, 128, 128, false, true);
        canvas.bake_ctx(&mut front, &ctx(&t, &pal, &treat, None, None));
        canvas.update_mask(&front, 2);
        assert!(canvas.caster_rects().len() >= 1, "a cold bake casts everywhere");

        // move one cell and rebake: the caster footprint must name that cell's tile
        front.set_mat(60, 60, 0);
        canvas.bake_ctx(&mut front, &ctx(&t, &pal, &treat, None, None));
        let rects = canvas.caster_rects();
        assert!(
            rects.iter().any(|r| (r.0..=r.2).contains(&60) && (r.1..=r.3).contains(&60)),
            "the changed cell is not in the caster footprint: {rects:?}"
        );
        // and the mask followed it
        canvas.update_mask(&front, 0);
        assert!(
            canvas.mask().sample(60.0 / MASK_DIV as f32, 60.0 / MASK_DIV as f32) < 1.0,
            "mask still reports that cell as fully solid"
        );
    }
}
