//! Angle-of-repose LUT. `repose_angle` is authored data; this maps it to the creep
//! rule's geometry and rate, calibrated headlessly against measured piles.

use crate::layer::Layer;
use crate::material::{MaterialClass, MaterialTable};
use crate::units::LayerSlot;

/// grains dropped per tick by the calibrator — 1 would make a 1200-grain trial 1200
/// ticks long, and the pile shape is unchanged by a 4-wide source
const CALIBRATE_SPAWN_PER_TICK: u16 = 4;
/// how many creep-geometry candidates the calibrator is allowed to try per angle
const CALIBRATE_ITERS: u32 = 7;
/// how long the calibrator waits for a pile to stop moving after the last grain
const CALIBRATE_SETTLE_TICKS: u64 = 4000;
/// consecutive zero-move ticks the calibrator requires before it calls a pile settled.
/// One quiet tick is not enough: creep is probabilistic, so a pile mid-avalanche can
/// have a quiet tick and then keep relaxing.
const CALIBRATE_QUIET_TICKS: u32 = 90;
/// floor on `p_creep` below 45 deg. The probability is a relaxation *rate*, not the
/// angle, and a rate near zero means a pile never reaches its geometric repose.
const CREEP_RATE_FLOOR: u8 = 96;

/// widest horizontal probe the creep rule may use, in cells
pub const MAX_CREEP_SPAN: u16 = 12;

pub struct ReposeLut {
    /// `p_creep * 255`, index = target angle in whole degrees (angle < 45)
    creep: [u8; 91],
    /// `p_stick * 255`, index = target angle (angle > 45)
    stick: [u8; 91],
    /// horizontal probe distance of the creep rule, in cells (1..=MAX_CREEP_SPAN)
    span: [u8; 91],
    /// surface drop, in cells, that the probe must see before a grain creeps
    drop: [u8; 91],
}

#[inline]
fn key(angle_deg: f32) -> usize {
    if angle_deg.is_nan() {
        return 45;
    }
    angle_deg.round().clamp(0.0, 90.0) as usize
}

impl ReposeLut {
    /// analytic seed curves; instant, used at startup
    ///
    /// The *geometry* (`span`/`drop`) sets the angle a pile relaxes to, because a
    /// probability alone cannot: any non-zero probability eventually flattens a slope
    /// unless the rule stops firing, so the rule needs a geometric stop condition.
    /// `span`/`drop` is the rational approximation of `tan(angle)` closest to the
    /// target with `span <= MAX_CREEP_SPAN`, which lands every authored angle within
    /// ~0.4 deg before a single grain is dropped; `calibrate` then measures real piles
    /// and keeps whichever nearby candidate actually reads closest.
    pub fn analytic() -> Self {
        let mut creep = [0u8; 91];
        let mut stick = [0u8; 91];
        let mut span = [1u8; 91];
        let mut drop = [1u8; 91];
        for a in 0..=90usize {
            let af = a as f32;
            let c = (((45.0 - af) / 45.0) * 0.9).clamp(0.0, 0.9);
            let s = ((af - 45.0) / 40.0).clamp(0.0, 0.95);
            creep[a] = if af < 45.0 {
                ((c * 255.0).round() as u8).max(CREEP_RATE_FLOOR)
            } else {
                0
            };
            stick[a] = (s * 255.0).round() as u8;
            let (sp, dr) = best_ratio(af);
            span[a] = sp;
            drop[a] = dr;
        }
        Self {
            creep,
            stick,
            span,
            drop,
        }
    }

    #[inline]
    pub fn p_creep(&self, angle_deg: f32) -> u8 {
        self.creep[key(angle_deg)]
    }

    #[inline]
    pub fn p_stick(&self, angle_deg: f32) -> u8 {
        self.stick[key(angle_deg)]
    }

    /// horizontal probe distance the creep rule uses at this target angle, in cells
    #[inline]
    pub fn creep_span(&self, angle_deg: f32) -> u16 {
        self.span[key(angle_deg)] as u16
    }

    /// surface drop the probe must see before a grain creeps, in cells
    #[inline]
    pub fn creep_drop(&self, angle_deg: f32) -> u16 {
        self.drop[key(angle_deg)] as u16
    }

    /// Headless calibration: for each distinct authored angle, drop `grains` from a
    /// point source, linear-fit the pile surface, and keep the creep geometry whose
    /// *measured* angle is closest to the authored one.
    ///
    /// The search runs over the `drop / span` candidates ordered by how close
    /// `atan(drop / span)` is to the target, stopping as soon as a candidate lands
    /// within 0.5 deg. It does **not** bisect `p_creep`: a probability cannot set a
    /// static angle of repose, because any non-zero probability keeps relaxing a slope
    /// until the rule stops firing. The geometry is the stop condition and therefore
    /// the angle; `p_creep` is the relaxation rate.
    pub fn calibrate(&mut self, table: &MaterialTable, angles: &[f32], grains: u32, seed: u64) {
        let mut done: Vec<usize> = Vec::new();
        for &angle in angles {
            let k = key(angle);
            if done.contains(&k) {
                continue;
            }
            done.push(k);
            // above 45 deg the pile is held by p_stick, which the diagonal rule
            // already seeds correctly; only the creep branch is calibrated here
            if !(angle > 0.0 && angle < 45.0) {
                continue;
            }
            let Some(mat) = pick_powder(table, angle) else {
                continue;
            };

            let mut best = (self.span[k], self.drop[k]);
            let mut best_err = f32::INFINITY;
            for (s, d) in ratio_candidates(angle) {
                let measured = self.trial(table, mat, k, s, d, grains, seed);
                let err = (measured - angle).abs();
                if err < best_err {
                    best_err = err;
                    best = (s, d);
                }
                if err <= 0.5 {
                    break;
                }
            }
            self.span[k] = best.0;
            self.drop[k] = best.1;
        }
    }

    /// one drop with `span[k] / drop[k]` overridden, returns the measured pile angle
    fn trial(
        &self,
        table: &MaterialTable,
        mat: u8,
        k: usize,
        span: u8,
        drop: u8,
        grains: u32,
        seed: u64,
    ) -> f32 {
        let mut probe = Self {
            creep: self.creep,
            stick: self.stick,
            span: self.span,
            drop: self.drop,
        };
        probe.span[k] = span;
        probe.drop[k] = drop;
        let layer = drop_pile(table, &probe, mat, grains, seed);
        Self::measure_pile_angle(&layer)
    }

    /// Least-squares fit of the surface height of each flank; the reported angle is
    /// the mean of the two flanks' `atan(|dh/dx|)`.
    pub fn measure_pile_angle(layer: &Layer) -> f32 {
        let w = layer.w as usize;
        let h = layer.h as usize;
        if w < 8 {
            return 0.0;
        }
        // surface height of every column, measured up from the floor row
        let mut height = vec![0u16; w];
        for x in 0..w {
            for y in 0..h {
                if layer.mat[y * w + x] != 0 {
                    height[x] = (h - y) as u16;
                    break;
                }
            }
        }
        let Some(apex) = (0..w).max_by_key(|&x| height[x]) else {
            return 0.0;
        };
        if height[apex] < 4 {
            return 0.0;
        }

        let mut sum = 0.0f32;
        let mut n = 0u32;
        // right flank, then left; skip 2 columns either side of the apex (the source
        // column is rounded) and the last 2 of the toe (single-grain noise)
        for dir in [1i32, -1] {
            let mut pts: Vec<(f32, f32)> = Vec::new();
            let mut x = apex as i32 + dir * 2;
            while x >= 0 && (x as usize) < w && height[x as usize] >= 2 {
                pts.push((x as f32, height[x as usize] as f32));
                x += dir;
            }
            if pts.len() > 4 {
                pts.truncate(pts.len() - 2);
            }
            if pts.len() < 3 {
                continue;
            }
            let m = pts.len() as f32;
            let mx = pts.iter().map(|p| p.0).sum::<f32>() / m;
            let my = pts.iter().map(|p| p.1).sum::<f32>() / m;
            let mut num = 0.0;
            let mut den = 0.0;
            for &(px, py) in &pts {
                num += (px - mx) * (py - my);
                den += (px - mx) * (px - mx);
            }
            if den <= 0.0 {
                continue;
            }
            sum += (num / den).abs().atan().to_degrees();
            n += 1;
        }
        if n == 0 { 0.0 } else { sum / n as f32 }
    }
}

/// The `drop / span` ratio whose `atan` is closest to `target`, `span <= MAX_CREEP_SPAN`.
fn best_ratio(target: f32) -> (u8, u8) {
    ratio_candidates(target)[0]
}

/// Up to [`CALIBRATE_ITERS`] `(span, drop)` pairs in increasing distance of
/// `atan(drop / span)` from `target`. Duplicated ratios (2/4 next to 1/2) are dropped,
/// so every candidate is a distinct angle.
fn ratio_candidates(target: f32) -> Vec<(u8, u8)> {
    let mut all: Vec<(f32, u8, u8)> = Vec::new();
    for s in 1..=MAX_CREEP_SPAN as u8 {
        for d in 1..=s {
            let a = (d as f32 / s as f32).atan().to_degrees();
            if all.iter().any(|&(b, _, _)| (b - a).abs() < 0.01) {
                continue;
            }
            all.push((a, s, d));
        }
    }
    all.sort_by(|x, y| {
        (x.0 - target)
            .abs()
            .partial_cmp(&(y.0 - target).abs())
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    all.truncate(CALIBRATE_ITERS as usize);
    all.into_iter().map(|(_, s, d)| (s, d)).collect()
}

/// first powder in the table whose dry repose angle matches `angle`
fn pick_powder(table: &MaterialTable, angle: f32) -> Option<u8> {
    table.iter().find_map(|(id, m)| {
        (m.class == MaterialClass::Powder && (m.repose_angle - angle).abs() < 0.5).then_some(id.0)
    })
}

/// Headless point-source pile drop. No floor material is needed: the bottom row of
/// the layer is the floor, because a grain at `h - 1` has nowhere to fall.
pub fn drop_pile(table: &MaterialTable, lut: &ReposeLut, mat: u8, grains: u32, seed: u64) -> Layer {
    let root = (grains as f32).sqrt();
    let w = ((root * 3.5) as u16 + 24).clamp(64, 1024);
    let h = ((root * 1.3) as u16 + 24).clamp(64, 512);
    let mut layer = Layer::new(w, h, LayerSlot::Plant, seed);

    let cx = w / 2;
    let mut left = grains;
    let mut tick = 0u64;
    let mut quiet = 0u32;
    let deadline = (grains as u64 / CALIBRATE_SPAWN_PER_TICK as u64) + CALIBRATE_SETTLE_TICKS;

    while tick < deadline {
        for k in 0..CALIBRATE_SPAWN_PER_TICK {
            if left == 0 {
                break;
            }
            let x = cx + k - CALIBRATE_SPAWN_PER_TICK / 2;
            if layer.mat_at(x, 0) == 0 {
                layer.set_mat(x, 0, mat);
                left -= 1;
            }
        }

        layer.tick = tick;
        layer.stats.reset();
        layer.chunks.begin_tick();
        layer.clear_moved_flags();
        crate::powder::step_powder(&mut layer, table, lut);
        let moved = layer.stats.moves;
        layer.chunks.end_tick();
        tick += 1;

        if moved == 0 {
            quiet += 1;
        } else {
            quiet = 0;
        }
        if left == 0 && quiet >= CALIBRATE_QUIET_TICKS {
            break;
        }
    }
    layer
}
