use std::marker::PhantomData;

use crate::{action::Action, state::StateKey};

#[derive(Clone, Debug)]
pub struct Plan<K: StateKey, A: Action<K>> {
    pub steps: Vec<A>,
    _phantom: PhantomData<K>,
}

impl<K: StateKey, A: Action<K>> Plan<K, A> {
    pub fn new(steps: Vec<A>) -> Self {
        Self {
            steps,
            _phantom: PhantomData,
        }
    }
}
