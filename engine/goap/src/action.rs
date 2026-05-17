use std::marker::PhantomData;

use crate::state::{Effect, Precondition, StateKey, WorldState};
use common::prelude::*;

pub struct PlanContext;

pub trait Action<K: StateKey>: Clone + Send + Sync + PartialEq + 'static {
    fn preconditions(&self) -> Precondition<K>;
    fn effects(&self) -> Effect<K>;
    fn cost(&self, state: &WorldState<K>, ctx: &PlanContext) -> f32 {
        1.0
    }
}
