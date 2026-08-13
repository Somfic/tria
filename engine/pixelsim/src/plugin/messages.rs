use common::prelude::*;

use crate::cell::CellRect;
use crate::material::MaterialId;

#[derive(Message)]
pub struct LayerEdited {
    pub layer: usize,
    pub rect: CellRect,
}

#[derive(Message)]
pub struct PortTransferred {
    pub port: usize,
    pub mat: MaterialId,
}
