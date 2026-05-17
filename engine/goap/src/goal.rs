use std::marker::PhantomData;

use crate::state::{State, StateKey};
use common::prelude::*;

#[derive(Clone, Debug)]
pub struct Goal<K: StateKey> {
    name: &'static str,
    pattern: State,
    mask: State,
    priority: f32,
    _phantom: PhantomData<K>,
}

impl<K: StateKey> Goal<K> {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            pattern: 0,
            mask: 0,
            priority: 1.0,
            _phantom: PhantomData,
        }
    }

    pub fn wants(mut self, key: K, value: bool) -> Self {
        self.mask |= key.bit();
        if value {
            self.pattern |= key.bit();
        }
        self
    }

    pub fn with_priority(mut self, p: f32) -> Self {
        self.priority = p;
        self
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn pattern(&self) -> u64 {
        self.pattern
    }

    pub fn mask(&self) -> u64 {
        self.mask
    }

    pub fn priority(&self) -> f32 {
        self.priority
    }
}
