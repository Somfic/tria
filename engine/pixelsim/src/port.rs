//! The only module in the crate that sees more than one layer.

use crate::cell::{FLAG_MOVED, FLAG_PORT};
use crate::layer::{CH_FLAGS, Layer};
use crate::material::MaterialTable;

#[derive(Copy, Clone, Debug)]
pub struct PortPair {
    pub from_layer: u16,
    pub from: (u16, u16),
    pub to_layer: u16,
    pub to: (u16, u16),
    /// cells moved per tick, 1 in the slice
    pub per_tick: u8,
}

/// The ONLY function in the crate that sees more than one layer. Serial, post-solve.
/// Returns the number of cells transferred.
///
/// Transfers are all-or-nothing per cell and carry all nine channels, so mass across
/// a port is exactly conserved and a spike can check it.
pub fn apply_ports(layers: &mut [Layer], ports: &[PortPair], table: &MaterialTable) -> u32 {
    let mut moved = 0u32;
    for p in ports {
        let (fl, tl) = (p.from_layer as usize, p.to_layer as usize);
        if fl >= layers.len() || tl >= layers.len() {
            continue;
        }
        let (fx, fy) = p.from;
        let (tx, ty) = p.to;
        if !layers[fl].in_bounds(fx as i32, fy as i32)
            || !layers[tl].in_bounds(tx as i32, ty as i32)
        {
            continue;
        }

        for _ in 0..p.per_tick {
            let fi = layers[fl].idx(fx, fy);
            let m = layers[fl].mat[fi];
            if table.is_empty(m) || table.is_solid(m) {
                break;
            }
            let ti = layers[tl].idx(tx, ty);
            if !table.is_empty(layers[tl].mat[ti]) {
                break;
            }

            let mut cell = layers[fl].read_cell(fi);
            let src = &mut layers[fl];
            src.mat[fi] = 0;
            src.flags[fi] = 0;
            src.clear_aux(fi);
            src.touch(fx, fy);

            // the transferred cell has not "moved" in its new layer's tick, but it is
            // port-borne, which the renderer and the chute spike both want to see
            cell[CH_FLAGS] = (cell[CH_FLAGS] & !FLAG_MOVED) | FLAG_PORT;
            let dst = &mut layers[tl];
            dst.write_cell(ti, cell);
            dst.touch(tx, ty);
            moved += 1;
        }
    }
    moved
}
