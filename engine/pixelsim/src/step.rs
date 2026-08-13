//! Per-tick orchestration. The only module that touches `bevy_tasks`, and it takes
//! the pool as a parameter so the module itself stays plugin-free.

use crate::cell::Bitset;
use crate::layer::Layer;
use crate::material::MaterialTable;
use crate::port::{PortPair, apply_ports};
use crate::repose::ReposeLut;
use crate::stats::{LayerStats, SimStats};
use crate::units::{AUX_EVERY, CHUNK_PX, COARSE_EVERY};
use crate::{liquid, powder, slurry, wet};

/// One task per layer. Layers share nothing, so per-layer determinism is independent
/// of how the pool schedules them.
pub fn step_all(
    layers: &mut Vec<Layer>,
    ports: &[PortPair],
    table: &MaterialTable,
    lut: &ReposeLut,
    tick: u64,
    pool: &bevy::tasks::TaskPool,
) -> SimStats {
    use bevy::tasks::ParallelSliceMut;

    let t0 = std::time::Instant::now();
    let per_layer =
        layers.par_chunk_map_mut(pool, 1, |_, s| step_layer(&mut s[0], table, lut, tick));
    let step_ms = t0.elapsed().as_secs_f32() * 1000.0;

    finish(layers, ports, table, per_layer, step_ms)
}

/// Same contract as [`step_all`] without a task pool — used when `SimConfig.parallel`
/// is off and by the headless tests, which must not spin up threads.
pub fn step_all_serial(
    layers: &mut Vec<Layer>,
    ports: &[PortPair],
    table: &MaterialTable,
    lut: &ReposeLut,
    tick: u64,
) -> SimStats {
    let t0 = std::time::Instant::now();
    let per_layer: Vec<LayerStats> = layers
        .iter_mut()
        .map(|l| step_layer(l, table, lut, tick))
        .collect();
    let step_ms = t0.elapsed().as_secs_f32() * 1000.0;

    finish(layers, ports, table, per_layer, step_ms)
}

fn finish(
    layers: &mut [Layer],
    ports: &[PortPair],
    table: &MaterialTable,
    per_layer: Vec<LayerStats>,
    step_ms: f32,
) -> SimStats {
    let tp = std::time::Instant::now();
    let transfers = apply_ports(layers, ports, table);
    let port_ms = tp.elapsed().as_secs_f32() * 1000.0;

    let mut out = SimStats::default();
    for l in &per_layer {
        out.accumulate(l);
    }
    out.step_ms = step_ms + port_ms;
    out.port_ms = port_ms;
    out.transfers = transfers;
    out
}

pub fn step_layer(
    layer: &mut Layer,
    table: &MaterialTable,
    lut: &ReposeLut,
    tick: u64,
) -> LayerStats {
    let t0 = std::time::Instant::now();

    // `One Simulated Plane`: exactly one slab runs a solver. The rest are static structure —
    // real geometry the player builds, walks on and routes through, but which nothing ever
    // moves. Returning here rather than filtering at the call site means *no* caller can
    // accidentally re-animate a static slab, and it keeps the per-layer stats shape intact so
    // the HUD still lists every slab (at 0 ms, which is the point).
    if !layer.slot.simulated() {
        layer.tick = tick;
        layer.stats.reset();
        // A static slab still has to publish dirty tiles once, or the renderer would never
        // bake it at all. `begin_tick` is cheap and leaves the chunk state coherent.
        layer.chunks.begin_tick();
        layer.stats.awake_chunks = 0;
        layer.stats.step_ms = t0.elapsed().as_secs_f32() * 1000.0;
        return layer.stats.clone();
    }

    layer.tick = tick;
    layer.stats.reset();
    let tb = std::time::Instant::now();
    layer.chunks.begin_tick();
    layer.stats.awake_chunks = layer.chunks.awake_count();
    layer.clear_moved_flags();
    let mut book = tb.elapsed().as_secs_f32() * 1000.0;

    let t = std::time::Instant::now();
    liquid::compute_head(layer, table);
    layer.stats.head_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = std::time::Instant::now();
    powder::step_powder(layer, table, lut);
    layer.stats.powder_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = std::time::Instant::now();
    liquid::step_liquid(layer, table);
    layer.stats.liquid_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = std::time::Instant::now();
    wet::step_wet(layer, table, tick);
    // the slow rules run off the damp set, not the awake set, so they survive sleep
    if tick % AUX_EVERY == 0 {
        wet::step_wet_aux(layer, table, tick);
    }
    layer.stats.wet_ms = t.elapsed().as_secs_f32() * 1000.0;

    let t = std::time::Instant::now();
    slurry::step_slurry(layer, table, tick);
    layer.stats.slurry_ms = t.elapsed().as_secs_f32() * 1000.0;

    if tick % COARSE_EVERY == 0 {
        layer.coarse.decay();
    }

    let tb = std::time::Instant::now();
    layer.chunks.end_tick();
    layer.mark_slept();
    book += tb.elapsed().as_secs_f32() * 1000.0;
    layer.stats.book_ms = book;

    layer.stats.step_ms = t0.elapsed().as_secs_f32() * 1000.0;
    layer.stats.clone()
}

/// Inclusive `[first, last]` cell span of chunk index `c` along an axis of length
/// `len`. In `u32`, because `c * CHUNK_PX + CHUNK_PX` exceeds `u16::MAX` on the last
/// chunk of a maximal-length axis and the wrapped value indexes out of the layer.
#[inline]
pub(crate) fn chunk_span(c: u16, len: u16) -> (u16, u16) {
    let a = c as u32 * CHUNK_PX as u32;
    (a as u16, ((a + CHUNK_PX as u32).min(len as u32) - 1) as u16)
}

/// The single iteration driver shared by powder/liquid/wet/slurry: awake chunks,
/// bottom chunk row first, `y` descending, `x` direction alternating on `tick`.
///
/// The active bitset is lifted out of the layer for the duration of the scan so the
/// callback can hold `&mut Layer`; nothing in a solver pass writes `active`, so this
/// is only a borrow trick, not a semantic one.
#[inline]
pub(crate) fn for_awake_cells_bottom_up(
    layer: &mut Layer,
    mut f: impl FnMut(&mut Layer, u16, u16),
) {
    let active = core::mem::replace(&mut layer.chunks.active, Bitset::new(0));
    let (cw, ch) = (layer.chunks.cw, layer.chunks.ch);
    let (w, h) = (layer.w, layer.h);
    let ltr = layer.tick & 1 == 0;

    for cy in (0..ch).rev() {
        let (y0, y1) = chunk_span(cy, h);
        for y in (y0..=y1).rev() {
            for ci in 0..cw {
                let cx = if ltr { ci } else { cw - 1 - ci };
                if !active.get(cy as usize * cw as usize + cx as usize) {
                    continue;
                }
                let (x0, x1) = chunk_span(cx, w);
                if ltr {
                    for x in x0..=x1 {
                        f(layer, x, y);
                    }
                } else {
                    for x in (x0..=x1).rev() {
                        f(layer, x, y);
                    }
                }
            }
        }
    }

    layer.chunks.active = active;
}
