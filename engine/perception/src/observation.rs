use common::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum SenseKind {
    Vision,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub subject: Entity,
    pub sense: SenseKind,
    pub timestamp: f64,
    pub confidence: f32,
    pub position: Vec3,
}
