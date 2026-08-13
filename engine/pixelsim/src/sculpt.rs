//! Pure sculpting functions over a single `Layer`. Every one writes
//! `FLAG_PLAYER_PLACED`, wakes + dirties the touched rect, opens/commits an undo
//! stroke, and returns the affected rect for the caller to publish as `LayerEdited`.

use crate::cell::{CellRect, FLAG_PLAYER_PLACED, FLAG_SLEEPING};
use crate::layer::Layer;
use crate::material::{MaterialId, MaterialTable};
use crate::undo::UndoJournal;

/// flood fill is bounded to this many cells either side of the seed
pub const FILL_HALF: u16 = 64;

#[derive(Copy, Clone, Debug)]
pub enum Brush {
    Round { r: u8 },
    Square { r: u8 },
}

impl Brush {
    #[inline]
    pub fn radius(self) -> u8 {
        match self {
            Brush::Round { r } | Brush::Square { r } => r,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Tool {
    Paint(MaterialId),
    Erase,
    Dig,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Mirror {
    pub x: Option<u16>,
    pub y: Option<u16>,
}

/// `aux` = the 7 non-`mat`/`flags` channels, interleaved
pub struct Clipboard {
    pub w: u16,
    pub h: u16,
    pub mat: Vec<u8>,
    pub flags: Vec<u8>,
    pub aux: Vec<u8>,
}

// ---- shared machinery -------------------------------------------------------

#[inline]
fn reflect(axis: u16, v: u16, limit: u16) -> Option<u16> {
    let r = 2 * axis as i32 - v as i32;
    (r >= 0 && r < limit as i32 && r != v as i32).then_some(r as u16)
}

/// The point and its mirror images. At most four, no allocation.
#[inline]
fn mirror_points(m: Mirror, x: u16, y: u16, w: u16, h: u16) -> ([(u16, u16); 4], usize) {
    let mut out = [(x, y); 4];
    let mut n = 1usize;
    let mx = m.x.and_then(|ax| reflect(ax, x, w));
    let my = m.y.and_then(|ay| reflect(ay, y, h));
    if let Some(rx) = mx {
        out[n] = (rx, y);
        n += 1;
    }
    if let Some(ry) = my {
        out[n] = (x, ry);
        n += 1;
    }
    if let (Some(rx), Some(ry)) = (mx, my) {
        out[n] = (rx, ry);
        n += 1;
    }
    (out, n)
}

/// Rect covering the mirror images of `r` as well as `r` itself.
fn mirrored_rect(r: CellRect, m: Mirror, w: u16, h: u16) -> CellRect {
    let mut out = r;
    let corners = [(r.x0, r.y0), (r.x1, r.y1)];
    for (cx, cy) in corners {
        let (pts, n) = mirror_points(m, cx, cy, w, h);
        for &(px, py) in &pts[..n] {
            out = out.union(CellRect::point(px, py));
        }
    }
    out.clamped(w, h)
}

fn expand(r: CellRect, by: u16, w: u16, h: u16) -> CellRect {
    CellRect {
        x0: r.x0.saturating_sub(by),
        y0: r.y0.saturating_sub(by),
        x1: r.x1.saturating_add(by),
        y1: r.y1.saturating_add(by),
    }
    .clamped(w, h)
}

/// Applies `tool` to exactly one cell. This is the only place a sculpt write happens,
/// so `FLAG_PLAYER_PLACED` cannot be forgotten.
#[inline]
fn tool_cell(l: &mut Layer, table: &MaterialTable, tool: Tool, x: u16, y: u16) {
    let i = l.idx(x, y);
    match tool {
        Tool::Paint(m) => {
            l.mat[i] = m.0;
            l.clear_aux(i);
        }
        Tool::Erase => {
            l.mat[i] = 0;
            l.clear_aux(i);
        }
        Tool::Dig => {
            if !dig_cost(table, l.mat[i]).is_finite() {
                return;
            }
            l.mat[i] = 0;
            l.clear_aux(i);
            // the freshly exposed face below shows the tool marks
            if y + 1 < l.h {
                let bi = l.idx(x, y + 1);
                l.wear[bi] = l.wear[bi].saturating_add(32);
            }
        }
    }
    l.flags[i] = (l.flags[i] & !FLAG_SLEEPING) | FLAG_PLAYER_PLACED;
}

/// Stamps the brush footprint centred on `(cx, cy)`.
fn stamp_at(l: &mut Layer, table: &MaterialTable, tool: Tool, brush: Brush, cx: u16, cy: u16) {
    let r = brush.radius() as i32;
    let round = matches!(brush, Brush::Round { .. });
    let rr = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            if round && dx * dx + dy * dy > rr {
                continue;
            }
            let x = cx as i32 + dx;
            let y = cy as i32 + dy;
            if !l.in_bounds(x, y) {
                continue;
            }
            tool_cell(l, table, tool, x as u16, y as u16);
        }
    }
}

fn stamp_mirrored(
    l: &mut Layer,
    table: &MaterialTable,
    tool: Tool,
    brush: Brush,
    x: u16,
    y: u16,
    mirror: Mirror,
) {
    let (pts, n) = mirror_points(mirror, x, y, l.w, l.h);
    for &(px, py) in &pts[..n] {
        stamp_at(l, table, tool, brush, px, py);
    }
}

fn finish(l: &mut Layer, journal: &mut UndoJournal, rect: CellRect) -> CellRect {
    journal.commit();
    l.chunks.wake_rect(rect);
    l.chunks.mark_dirty_rect(rect);
    rect
}

// ---- public tools -----------------------------------------------------------

pub fn apply_brush(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    table: &MaterialTable,
    at: (u16, u16),
    brush: Brush,
    tool: Tool,
    mirror: Mirror,
) -> CellRect {
    let base = mirrored_rect(CellRect::point(at.0, at.1), mirror, layer.w, layer.h);
    let rect = expand(base, brush.radius() as u16, layer.w, layer.h);
    journal.begin(layer, layer_index, rect);
    stamp_mirrored(layer, table, tool, brush, at.0, at.1, mirror);
    finish(layer, journal, rect)
}

pub fn apply_line(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    table: &MaterialTable,
    a: (u16, u16),
    b: (u16, u16),
    brush: Brush,
    tool: Tool,
    mirror: Mirror,
) -> CellRect {
    let span = CellRect {
        x0: a.0,
        y0: a.1,
        x1: b.0,
        y1: b.1,
    }
    .clamped(layer.w, layer.h);
    let base = mirrored_rect(span, mirror, layer.w, layer.h);
    let rect = expand(base, brush.radius() as u16, layer.w, layer.h);
    journal.begin(layer, layer_index, rect);
    for_line(a, b, |x, y| {
        stamp_mirrored(layer, table, tool, brush, x, y, mirror);
    });
    finish(layer, journal, rect)
}

pub fn apply_rect(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    table: &MaterialTable,
    r: CellRect,
    filled: bool,
    tool: Tool,
    mirror: Mirror,
) -> CellRect {
    let span = r.clamped(layer.w, layer.h);
    let rect = mirrored_rect(span, mirror, layer.w, layer.h);
    journal.begin(layer, layer_index, rect);
    for y in span.y0..=span.y1 {
        for x in span.x0..=span.x1 {
            if !filled && x != span.x0 && x != span.x1 && y != span.y0 && y != span.y1 {
                continue;
            }
            let (pts, n) = mirror_points(mirror, x, y, layer.w, layer.h);
            for &(px, py) in &pts[..n] {
                tool_cell(layer, table, tool, px, py);
            }
        }
    }
    finish(layer, journal, rect)
}

/// Quadratic Bezier `a -> c` with control point `b`; the control-point bounding box
/// contains the curve, so the undo rect is exact without sampling twice.
pub fn apply_arc(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    table: &MaterialTable,
    a: (u16, u16),
    b: (u16, u16),
    c: (u16, u16),
    brush: Brush,
    tool: Tool,
    mirror: Mirror,
) -> CellRect {
    let hull = CellRect {
        x0: a.0.min(b.0).min(c.0),
        y0: a.1.min(b.1).min(c.1),
        x1: a.0.max(b.0).max(c.0),
        y1: a.1.max(b.1).max(c.1),
    }
    .clamped(layer.w, layer.h);
    let base = mirrored_rect(hull, mirror, layer.w, layer.h);
    let rect = expand(base, brush.radius() as u16, layer.w, layer.h);
    journal.begin(layer, layer_index, rect);

    let steps = (hull.width() as u32 + hull.height() as u32).max(2) * 2;
    let mut last: Option<(u16, u16)> = None;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let u = 1.0 - t;
        let px = u * u * a.0 as f32 + 2.0 * u * t * b.0 as f32 + t * t * c.0 as f32;
        let py = u * u * a.1 as f32 + 2.0 * u * t * b.1 as f32 + t * t * c.1 as f32;
        let p = (
            px.round().clamp(0.0, (layer.w - 1) as f32) as u16,
            py.round().clamp(0.0, (layer.h - 1) as f32) as u16,
        );
        if last == Some(p) {
            continue;
        }
        last = Some(p);
        stamp_mirrored(layer, table, tool, brush, p.0, p.1, mirror);
    }
    finish(layer, journal, rect)
}

/// bounded to 128x128 around `at`
pub fn flood_fill(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    table: &MaterialTable,
    at: (u16, u16),
    mat: MaterialId,
) -> CellRect {
    let rect = expand(CellRect::point(at.0, at.1), FILL_HALF, layer.w, layer.h);
    if !layer.in_bounds(at.0 as i32, at.1 as i32) {
        return rect;
    }
    let target = layer.mat_at(at.0, at.1);
    if target == mat.0 {
        return rect;
    }
    journal.begin(layer, layer_index, rect);

    // scanline-free BFS over the bounded window; one Vec per call, never per cell
    let mut stack: Vec<(u16, u16)> = Vec::with_capacity(256);
    stack.push(at);
    while let Some((x, y)) = stack.pop() {
        if !rect.contains(x, y) || layer.mat_at(x, y) != target {
            continue;
        }
        tool_cell(layer, table, Tool::Paint(mat), x, y);
        if x > rect.x0 {
            stack.push((x - 1, y));
        }
        if x < rect.x1 {
            stack.push((x + 1, y));
        }
        if y > rect.y0 {
            stack.push((x, y - 1));
        }
        if y < rect.y1 {
            stack.push((x, y + 1));
        }
    }
    finish(layer, journal, rect)
}

pub fn copy_region(layer: &Layer, r: CellRect) -> Clipboard {
    let r = r.clamped(layer.w, layer.h);
    let (w, h) = (r.width(), r.height());
    let n = w as usize * h as usize;
    let mut clip = Clipboard {
        w,
        h,
        mat: Vec::with_capacity(n),
        flags: Vec::with_capacity(n),
        aux: Vec::with_capacity(n * 7),
    };
    for y in r.y0..=r.y1 {
        for x in r.x0..=r.x1 {
            let c = layer.read_cell(layer.idx(x, y));
            clip.mat.push(c[0]);
            clip.flags.push(c[1]);
            clip.aux.extend_from_slice(&c[2..9]);
        }
    }
    clip
}

pub fn paste_region(
    layer: &mut Layer,
    journal: &mut UndoJournal,
    layer_index: u16,
    clip: &Clipboard,
    at: (u16, u16),
    mirror: Mirror,
) -> CellRect {
    let span = CellRect {
        x0: at.0,
        y0: at.1,
        x1: at.0.saturating_add(clip.w.saturating_sub(1)),
        y1: at.1.saturating_add(clip.h.saturating_sub(1)),
    }
    .clamped(layer.w, layer.h);
    let rect = mirrored_rect(span, mirror, layer.w, layer.h);
    if clip.w == 0 || clip.h == 0 {
        return rect;
    }
    journal.begin(layer, layer_index, rect);

    for cy in 0..clip.h {
        for cx in 0..clip.w {
            let k = cy as usize * clip.w as usize + cx as usize;
            let mut cell = [0u8; 9];
            cell[0] = clip.mat[k];
            cell[1] = clip.flags[k] | FLAG_PLAYER_PLACED;
            cell[2..9].copy_from_slice(&clip.aux[k * 7..k * 7 + 7]);
            cell[1] &= !FLAG_SLEEPING;

            let x = at.0 as i32 + cx as i32;
            let y = at.1 as i32 + cy as i32;
            if !layer.in_bounds(x, y) {
                continue;
            }
            let (pts, n) = mirror_points(mirror, x as u16, y as u16, layer.w, layer.h);
            for &(px, py) in &pts[..n] {
                let i = layer.idx(px, py);
                layer.write_cell(i, cell);
            }
        }
    }
    finish(layer, journal, rect)
}

/// proportional to hardness; `INFINITY` at `>= 10`
pub fn dig_cost(table: &MaterialTable, mat: u8) -> f32 {
    let hd = table.hardness(mat);
    if hd >= 10.0 {
        f32::INFINITY
    } else {
        hd.max(0.1)
    }
}

/// Bresenham, so a dragged stroke has no gaps at any slope.
fn for_line(a: (u16, u16), b: (u16, u16), mut f: impl FnMut(u16, u16)) {
    let (mut x0, mut y0) = (a.0 as i32, a.1 as i32);
    let (x1, y1) = (b.0 as i32, b.1 as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        f(x0 as u16, y0 as u16);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
