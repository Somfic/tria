use common::prelude::*;
use std::collections::HashMap;

use crate::Observation;

#[derive(Clone, Copy, Debug)]
pub struct MemoryConfig {
    pub decay_per_second: f32,
    pub forgotten_threshold: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub observation: Observation,
}

#[derive(Component, Clone, Debug)]
pub struct PerceptionMemory {
    config: MemoryConfig,
    memories: HashMap<Entity, Memory>,
}

impl PerceptionMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            config,
            memories: HashMap::new(),
        }
    }

    pub fn record(&mut self, observation: Observation) {
        let memory = self.memories.get(&observation.subject);

        match memory {
            Some(existing) => {
                if observation.confidence > existing.observation.confidence {
                    self.memories
                        .insert(observation.subject, Memory { observation });
                }
            }
            None => {
                self.memories
                    .insert(observation.subject, Memory { observation });
            }
        }
    }

    pub fn knows_about(&self, entity: Entity) -> bool {
        self.memories.contains_key(&entity)
    }

    pub fn confidence(&self, entity: Entity) -> f32 {
        self.memories
            .get(&entity)
            .map(|memory| memory.observation.confidence)
            .unwrap_or(0.0)
    }

    pub fn position(&self, entity: Entity) -> Option<Vec3> {
        self.memories
            .get(&entity)
            .map(|memory| memory.observation.position)
    }

    pub fn time_since_observed(&self, entity: Entity, now: f64) -> Option<f64> {
        self.memories
            .get(&entity)
            .map(|memory| now - memory.observation.timestamp)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Entity, &Memory)> {
        self.memories.iter()
    }

    pub fn decay(&mut self, dt: f32) {
        let decay_amount = self.config.decay_per_second * dt;
        self.memories.retain(|_, memory| {
            let new_confidence = memory.observation.confidence - decay_amount;
            new_confidence > self.config.forgotten_threshold
        });
    }
}
