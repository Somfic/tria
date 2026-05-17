use crate::{
    action::Action,
    goal::Goal,
    plan::Plan,
    state::{StateKey, WorldState},
};
use common::prelude::*;

#[derive(Component)]
pub struct PerceivedState<K: StateKey>(pub WorldState<K>);

#[derive(Component)]
pub struct CurrentGoal<K: StateKey>(pub Goal<K>);

#[derive(Component)]
pub struct CurrentPlan<K: StateKey, A: Action<K>> {
    pub plan: Plan<K, A>,
    pub step_index: usize,
    pub step_started: bool,
}

#[derive(Component)]
pub struct ActionSet<K: StateKey, A: Action<K>> {
    pub actions: Vec<A>,
    _phantom: PhantomData<K>,
}

impl<K: StateKey, A: Action<K>> ActionSet<K, A> {
    pub fn new(actions: Vec<A>) -> Self {
        Self {
            actions,
            _phantom: PhantomData,
        }
    }
}

#[derive(Component)]
pub struct GoapAgent;
