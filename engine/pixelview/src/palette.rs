//! 256-entry palette baked from the material table, plus aux-channel blending.
//!
//! Nothing here knows about layers or treatments: this is "what colour is this cell,
//! on its own terms". Depth readability happens later, in `treat`.

use pixelsim::{FLAG_SLEEPING, MaterialClass, MaterialId, MaterialTable};

pub struct Palette {
    pub rgb: [[u8; 3]; 256],
}

impl Palette {
    pub fn from_table(table: &MaterialTable) -> Self {
        // ids past the end of the table are a bug in the caller, not a colour choice —
        // make them scream rather than silently render as air-black
        let mut rgb = [MISSING_RGB; 256];
        for i in 0..table.len().min(256) {
            rgb[i] = table.get(MaterialId(i as u8)).color;
        }
        Self { rgb }
    }

    /// Composition order: base palette -> wetness darkening -> suspension lerp ->
    /// dirt lerp -> wear lighten.
    ///
    /// `flags` is consulted only for `FLAG_SLEEPING` (the dead-zone debug tint). The
    /// caller masks that bit off when the tint is disabled, so this needs no config
    /// parameter and stays branch-cheap.
    #[inline]
    pub fn cell_color(
        &self,
        mat: u8,
        flags: u8,
        wetness: u8,
        dirt: u8,
        wear: u8,
        susp_mat: u8,
        susp_conc: u8,
    ) -> [u8; 3] {
        let mut rgb = self.rgb[mat as usize];

        // wet material is darker material. the single most legible physical cue in the
        // sim, and it costs one multiply
        if wetness > 0 {
            let k = 1.0 - 0.35 * (wetness as f32 * INV255);
            rgb = [
                scale_u8(rgb[0], k),
                scale_u8(rgb[1], k),
                scale_u8(rgb[2], k),
            ];
        }

        // suspended solids colour the liquid carrying them — this is how a settling
        // pond reads as "still dirty" with no extra state
        if susp_conc > 0 {
            rgb = lerp_rgb(
                rgb,
                self.rgb[susp_mat as usize],
                susp_conc as f32 * INV255 * 0.8,
            );
        }

        // damp that dried leaves a stain ring; see the vault's Micro-interactions
        if dirt > 0 {
            rgb = lerp_rgb(rgb, STAIN_RGB, dirt as f32 * INV255 * 0.6);
        }

        // wear abrades toward white — a rim worn thin where it touches ground
        if wear > 0 {
            rgb = lerp_rgb(rgb, [255, 255, 255], wear as f32 * INV255 * 0.15);
        }

        // debug only: settled+sleeping powder is exactly the "dead zone in your funnel"
        // signal, so tinting it makes the Flow lens' job visible before the lens exists
        if flags & FLAG_SLEEPING != 0 {
            rgb = lerp_rgb(rgb, SLEEP_RGB, SLEEP_MIX);
        }

        rgb
    }
}

/// the colour a fully-stained cell tends toward
pub const STAIN_RGB: [u8; 3] = [54, 48, 42];
/// a material id with no table entry — deliberately impossible to mistake for content
pub const MISSING_RGB: [u8; 3] = [255, 0, 255];
/// dead-zone debug tint for sleeping cells
pub const SLEEP_RGB: [u8; 3] = [64, 96, 255];
/// how far a sleeping cell is pulled toward `SLEEP_RGB`
pub const SLEEP_MIX: f32 = 0.22;

const INV255: f32 = 1.0 / 255.0;

#[inline]
fn scale_u8(v: u8, k: f32) -> u8 {
    (v as f32 * k).clamp(0.0, 255.0) as u8
}

#[inline]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * t) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * t) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * t) as u8,
    ]
}

/// Stateless per-cell hash. Deliberately *not* `pixelsim::hash_rng`: the handmade
/// grain must be stable for the life of a cell and must not shift if the sim ever
/// re-tunes its own PRNG. splitmix64 finaliser over the packed tuple.
#[inline]
pub fn cell_hash(mat: u8, x: u16, y: u16) -> u32 {
    let mut z = (x as u64) | ((y as u64) << 16) | ((mat as u64) << 32);
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 32) as u32
}

/// Per-class grain: `(luma amplitude, coordinate right-shift)`.
///
/// The vault's art direction is "handmade cutaway ... visible stitching, glue, rivets,
/// patches", so one flat colour per material is the wrong answer — it reads as
/// programmer art. Each class gets its own *frequency* as well as amplitude, which is
/// what separates a heap of grains from a sheet of material:
///
/// * **Powder** — strongest, per-pixel, scaled by grain size, so grit visibly speckles
///   coarser than silt. The grains are the thing you are looking at.
/// * **Solid** — moderate, on a 2x2 block: weave/hammer/patch texture on cardboard or
///   tin reads at a coarser scale than a grain of sand.
/// * **Liquid** — weakest. A liquid body must read as one coherent volume; loud
///   per-pixel noise fights the surface, and the surface is the readable part.
/// * **Gas / empty** — none.
#[inline]
pub fn grain_of(table: &MaterialTable, mat: u8) -> (f32, u8) {
    match table.class(mat) {
        MaterialClass::Powder => {
            let coarse = (table.grain(mat) / 600.0).clamp(0.0, 1.0);
            (0.06 + 0.06 * coarse, 0)
        }
        MaterialClass::Solid => (0.055, 1),
        MaterialClass::Liquid => (0.028, 0),
        MaterialClass::Gas | MaterialClass::Empty => (0.0, 0),
    }
}

/// Apply the handmade grain: a symmetric luma jitter of +/-`amp`, quantised to
/// `1 << shift` pixel blocks and stable for the life of the cell.
#[inline]
pub fn speckle(rgb: [u8; 3], amp: f32, shift: u8, mat: u8, x: u16, y: u16) -> [u8; 3] {
    if amp <= 0.0 {
        return rgb;
    }
    speckle_byte(rgb, amp, grain_jitter(mat, shift, x, y))
}

/// The jitter byte [`speckle`] derives from a cell's coordinates — the only part of the
/// grain that depends on *where* a cell is.
///
/// Split out because the bake memoises the fully treated colour on `(material, jitter)`:
/// the grain has exactly 256 outcomes per material, so a tile of one material costs 256
/// `PixelTreat::apply` calls instead of one per pixel. The split lives here, next to
/// `speckle`, so the memo cannot drift from the grain it is memoising.
#[inline]
pub fn grain_jitter(mat: u8, shift: u8, x: u16, y: u16) -> u8 {
    (cell_hash(mat, x >> shift, y >> shift) & 0xff) as u8
}

/// [`speckle`] with the coordinate hash already reduced to its jitter byte.
#[inline]
pub fn speckle_byte(rgb: [u8; 3], amp: f32, jitter: u8) -> [u8; 3] {
    if amp <= 0.0 {
        return rgb;
    }
    // jitter/255 -> [0,1], recentre to [-1,1]
    let j = (jitter as f32 * INV255) * 2.0 - 1.0;
    let k = 1.0 + amp * j;
    [
        scale_u8(rgb[0], k),
        scale_u8(rgb[1], k),
        scale_u8(rgb[2], k),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The handmade grain must be stable for the life of a cell — if it changed per frame
    /// the whole field would crawl.
    #[test]
    fn cell_hash_is_stable_and_varies_per_cell() {
        assert_eq!(cell_hash(3, 10, 20), cell_hash(3, 10, 20));
        assert_ne!(cell_hash(3, 10, 20), cell_hash(3, 11, 20));
        assert_ne!(cell_hash(3, 10, 20), cell_hash(3, 10, 21));
        // and different materials in the same cell differ
        assert_ne!(cell_hash(3, 10, 20), cell_hash(4, 10, 20));
    }

    #[test]
    fn speckle_stays_within_amplitude_and_is_a_no_op_at_zero() {
        let base = [100u8, 100, 100];
        assert_eq!(speckle(base, 0.0, 0, 1, 5, 5), base);
        for x in 0..64u16 {
            for y in 0..64u16 {
                let out = speckle(base, 0.10, 0, 2, x, y);
                assert!(
                    out[0] >= 89 && out[0] <= 111,
                    "grain escaped +/-10%: {}",
                    out[0]
                );
            }
        }
    }

    /// The memo the bake keys on `(material, jitter)` is only sound if the two halves of
    /// the split reassemble into exactly `speckle`.
    #[test]
    fn speckle_byte_reassembles_speckle() {
        for amp in [0.0f32, 0.028, 0.055, 0.12] {
            for shift in [0u8, 1] {
                for mat in [1u8, 2, 7, 200] {
                    for (x, y) in [(0u16, 0u16), (3, 9), (511, 255), (1023, 4)] {
                        let base = [193u8, 121, 44];
                        assert_eq!(
                            speckle(base, amp, shift, mat, x, y),
                            speckle_byte(base, amp, grain_jitter(mat, shift, x, y)),
                        );
                    }
                }
            }
        }
    }

    /// Solids speckle on a 2x2 block (weave/patch texture), powders per pixel (grains).
    #[test]
    fn solid_grain_is_blocky_and_powder_grain_is_not() {
        let base = [128u8, 128, 128];
        // shift 1 -> the four cells of a block share a value
        let a = speckle(base, 0.06, 1, 7, 4, 4);
        let b = speckle(base, 0.06, 1, 7, 5, 5);
        assert_eq!(a, b);
        // shift 0 -> neighbours differ
        let c = speckle(base, 0.06, 0, 7, 4, 4);
        let d = speckle(base, 0.06, 0, 7, 5, 5);
        assert_ne!(c, d);
    }
}
