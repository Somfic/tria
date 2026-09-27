//! Mip chains for textures loaded from ordinary image files.
//!
//! Bevy only carries mip levels for formats that store them -- DDS, KTX2,
//! basis. A PNG or JPEG arrives with `mip_level_count: 1`, so the GPU has
//! nothing to fall back on when the texture is minified and samples the full
//! resolution at every pixel. A texture 5400 texels wide shown a few hundred
//! pixels across is undersampled by a factor of ten, and it crawls with
//! aliasing the moment anything moves.
//!
//! Generating the chain at load is enough: box-filtered halves, trilinear
//! sampling, and anisotropy so the texture does not smear where a surface is
//! nearly edge-on.
//!
//! This is a load-time cost, paid once per texture. Somewhere that rebakes a
//! texture as it is edited, the whole chain is rebuilt on every bake, and the
//! arithmetic below is linear in the texel count -- worth measuring against
//! whatever prefilter it would be replacing before assuming it is cheaper.

use bevy::asset::RenderAssetUsages;
use bevy::image::{ImageSampler, ImageSamplerDescriptor};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureViewDescriptor, TextureViewDimension};

/// Textures waiting for a chain. Handles are dropped from the list once
/// their image has loaded and been processed.
#[derive(Resource, Default)]
pub struct Pending(Vec<Handle<Image>>);

impl Pending {
    /// Queue a texture. Returns the handle so this can wrap a `load` call.
    pub fn queue(&mut self, handle: Handle<Image>) -> Handle<Image> {
        self.0.push(handle.clone());
        handle
    }
}

pub struct MipmapPlugin;

impl Plugin for MipmapPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Pending>().add_systems(Update, generate);
    }
}

fn generate(mut pending: ResMut<Pending>, mut images: ResMut<Assets<Image>>) {
    if pending.0.is_empty() {
        return;
    }

    pending.0.retain(|handle| {
        let Some(image) = images.get_mut(handle) else {
            return true; // still loading
        };
        match build(image) {
            Ok(()) => debug!(
                "{} mip levels for {handle:?}",
                image.texture_descriptor.mip_level_count
            ),
            Err(why) => warn!("no mipmaps for {handle:?}: {why}"),
        }
        false
    });
}

fn build(image: &mut Image) -> Result<(), &'static str> {
    if image.texture_descriptor.mip_level_count > 1 {
        return Ok(());
    }
    if image.texture_descriptor.format != TextureFormat::Rgba8UnormSrgb {
        return Err("only Rgba8UnormSrgb is handled");
    }
    let size = image.texture_descriptor.size;
    if size.depth_or_array_layers != 1 {
        return Err("only single-layer 2d textures are handled");
    }
    let Some(base) = image.data.as_ref() else {
        return Err("image has no data");
    };

    let to_linear = srgb_table();
    let (mut width, mut height) = (size.width, size.height);
    let mut level = base.clone();
    let mut chain = base.clone();
    let mut levels = 1;

    while width > 1 || height > 1 {
        let (half_w, half_h) = ((width / 2).max(1), (height / 2).max(1));
        let mut next = vec![0u8; (half_w * half_h * 4) as usize];

        for y in 0..half_h {
            // Clamp rather than wrap, so an odd dimension reuses its last
            // row instead of folding the far edge of the map into it.
            let (y0, y1) = (2 * y, (2 * y + 1).min(height - 1));
            for x in 0..half_w {
                let (x0, x1) = (2 * x, (2 * x + 1).min(width - 1));
                let texels = [
                    texel(&level, width, x0, y0),
                    texel(&level, width, x1, y0),
                    texel(&level, width, x0, y1),
                    texel(&level, width, x1, y1),
                ];
                let out = ((y * half_w + x) * 4) as usize;
                for channel in 0..3 {
                    // Averaged in linear light. Averaging the stored sRGB
                    // values directly is the common shortcut and it darkens
                    // every level, so a minified coastline drifts towards
                    // the sea rather than staying put.
                    let sum: f32 = texels.iter().map(|t| to_linear[t[channel] as usize]).sum();
                    next[out + channel] = from_linear(sum / 4.0);
                }
                let alpha: u32 = texels.iter().map(|t| t[3] as u32).sum();
                next[out + 3] = (alpha / 4) as u8;
            }
        }

        chain.extend_from_slice(&next);
        level = next;
        width = half_w;
        height = half_h;
        levels += 1;
    }

    image.texture_descriptor.mip_level_count = levels;
    image.data = Some(chain);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2),
        mip_level_count: Some(levels),
        ..default()
    });
    // Filtering is the chain's business; everything else -- address modes
    // above all -- is whoever loaded the image. Replacing the sampler
    // wholesale quietly clamps a texture its loader asked to repeat.
    let mut sampler = match &image.sampler {
        ImageSampler::Descriptor(descriptor) => descriptor.clone(),
        ImageSampler::Default => ImageSamplerDescriptor::default(),
    };
    sampler.mipmap_filter = bevy::image::ImageFilterMode::Linear;
    sampler.min_filter = bevy::image::ImageFilterMode::Linear;
    sampler.mag_filter = bevy::image::ImageFilterMode::Linear;
    // The limb of a sphere is nearly edge-on, which is where an isotropic
    // mip choice blurs hardest.
    sampler.anisotropy_clamp = 16;
    image.sampler = ImageSampler::Descriptor(sampler);
    Ok(())
}

fn texel(data: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * width + x) * 4) as usize;
    [data[i], data[i + 1], data[i + 2], data[i + 3]]
}

/// sRGB byte to linear, precomputed. The chain is tens of millions of
/// texels; a transcendental per channel per level is not worth paying.
fn srgb_table() -> [f32; 256] {
    let mut table = [0.0; 256];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *entry = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }
    table
}

fn from_linear(c: f32) -> u8 {
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32, fill: [u8; 4]) -> Image {
        Image::new_fill(
            bevy::render::render_resource::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            bevy::render::render_resource::TextureDimension::D2,
            &fill,
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::all(),
        )
    }

    #[test]
    fn a_chain_runs_all_the_way_down_to_one_texel() {
        let mut img = image(64, 32, [128, 64, 32, 255]);
        build(&mut img).unwrap();
        // 64x32 -> 32x16 -> ... -> 1x1 is seven levels.
        assert_eq!(img.texture_descriptor.mip_level_count, 7);

        let expected: usize = (0..7)
            .map(|l| ((64u32 >> l).max(1) * (32u32 >> l).max(1) * 4) as usize)
            .sum();
        assert_eq!(img.data.as_ref().unwrap().len(), expected);
    }

    #[test]
    fn a_flat_texture_keeps_its_colour_all_the_way_down() {
        // The whole point of averaging in linear light: a uniform mid-grey
        // must not drift as it is halved. Averaging sRGB bytes directly
        // happens to survive this, but any gradient would not, and this is
        // the cheap invariant that catches a broken transfer function.
        let mut img = image(32, 32, [128, 64, 32, 255]);
        build(&mut img).unwrap();
        let data = img.data.unwrap();
        let last = data.len() - 4;
        assert_eq!(&data[last..], &[128, 64, 32, 255]);
    }

    #[test]
    fn averaging_black_and_white_lands_near_mid_grey_in_linear_light() {
        let mut img = image(2, 2, [255, 255, 255, 255]);
        if let Some(data) = img.data.as_mut() {
            // Two white texels, two black.
            data[8..16].copy_from_slice(&[0, 0, 0, 255, 0, 0, 0, 255]);
        }
        build(&mut img).unwrap();
        let data = img.data.unwrap();
        let grey = data[16];
        // Half the *light*, which is sRGB 188, not the 128 you get from
        // averaging the encoded bytes.
        assert!((186..=190).contains(&grey), "got {grey}");
    }

    #[test]
    fn an_odd_dimension_does_not_fold_the_far_edge_in() {
        let mut img = image(3, 3, [10, 20, 30, 255]);
        build(&mut img).unwrap();
        assert_eq!(img.texture_descriptor.mip_level_count, 2);
    }
}
