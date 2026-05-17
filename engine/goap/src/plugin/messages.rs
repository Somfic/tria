use crate::{action::Action, state::StateKey};
use common::prelude::*;

#[derive(Message)]
pub struct ActionStartRequested<K: StateKey, A: Action<K>> {
    pub agent: Entity,
    pub action: A,
    _phantom: PhantomData<K>,
}

impl<K: StateKey, A: Action<K>> ActionStartRequested<K, A> {
    pub fn new(agent: Entity, action: A) -> Self {
        Self {
            agent,
            action,
            _phantom: PhantomData,
        }
    }
}

#[derive(Message)]
pub struct ActionCompleted<K: StateKey, A: Action<K>> {
    pub agent: Entity,
    pub action: A,
    pub success: bool,
    _phantom: PhantomData<K>,
}

impl<K: StateKey, A: Action<K>> ActionCompleted<K, A> {
    pub fn new(agent: Entity, action: A, success: bool) -> Self {
        Self {
            agent,
            action,
            success,
            _phantom: PhantomData,
        }
    }
}
