use common::prelude::*;

#[derive(Resource)]
pub struct PlanConfig {
    pub max_nodes: usize,
    pub max_plan_length: usize,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1000,
            max_plan_length: 16,
        }
    }
}
