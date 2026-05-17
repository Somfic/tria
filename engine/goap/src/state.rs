use std::{borrow::Borrow, fmt::Debug, hash::Hash, marker::PhantomData};

pub type State = u64;

pub trait StateKey: Send + Sync + Copy + Eq + Hash + 'static + Debug {
    fn bit_position(self) -> u8;
    fn bit(self) -> State {
        let pos = self.bit_position();
        debug_assert!(pos < 64, "StateKey bit position {} out of range", pos);
        1u64 << pos
    }
    fn all() -> &'static [Self];
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorldState<K: StateKey> {
    bits: State,
    _phantom: PhantomData<K>,
}

impl<K: StateKey> Debug for WorldState<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut list = f.debug_list();
        for key in K::all() {
            list.entry(&format_args!("{:?}={}", key, self.get(*key)));
        }
        list.finish()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Precondition<K: StateKey> {
    pub pattern: State,
    pub mask: State,
    _phantom: PhantomData<K>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Effect<K: StateKey> {
    pub pattern: State,
    pub mask: State,
    _phantom: PhantomData<K>,
}

impl<K: StateKey> Precondition<K> {
    pub fn new() -> Self {
        Self {
            pattern: 0,
            mask: 0,
            _phantom: PhantomData,
        }
    }
    pub fn requires(mut self, key: impl Borrow<K>, value: bool) -> Self {
        let key = key.borrow();
        self.mask |= key.bit();
        if value {
            self.pattern |= key.bit();
        }
        self
    }
}

impl<K: StateKey> Effect<K> {
    pub fn new() -> Self {
        Self {
            pattern: 0,
            mask: 0,
            _phantom: PhantomData,
        }
    }

    pub fn sets(mut self, key: impl Borrow<K>, value: bool) -> Self {
        let key = key.borrow();
        self.mask |= key.bit();
        if value {
            self.pattern |= key.bit();
        }
        self
    }
}

impl<K: StateKey> WorldState<K> {
    pub fn new() -> Self {
        Self {
            bits: 0,
            _phantom: PhantomData,
        }
    }

    pub fn get(&self, key: impl Borrow<K>) -> bool {
        let key = key.borrow();
        (self.bits & key.bit()) != 0
    }

    pub fn set(&mut self, key: impl Borrow<K>, value: bool) {
        let key = key.borrow();
        if value {
            self.bits |= key.bit();
        } else {
            self.bits &= !key.bit();
        }
    }

    pub fn satisfies(&self, precondition: &Precondition<K>) -> bool {
        (self.bits & precondition.mask) == (precondition.pattern & precondition.mask)
    }

    pub fn apply(&self, effect: &Effect<K>) -> Self {
        Self {
            bits: (self.bits & !effect.mask) | (effect.pattern & effect.mask),
            _phantom: PhantomData,
        }
    }

    pub fn matches(&self, pattern: State, mask: State) -> bool {
        (self.bits & mask) == (pattern & mask)
    }

    pub fn distance_to(&self, pattern: State, mask: State) -> u32 {
        ((self.bits ^ pattern) & mask).count_ones()
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
    #[repr(u8)]
    enum TestKey {
        A = 0,
        B = 1,
        C = 2,
    }

    impl StateKey for TestKey {
        fn bit_position(self) -> u8 {
            self as u8
        }
        fn all() -> &'static [Self] {
            &[Self::A, Self::B, Self::C]
        }
    }

    #[test]
    fn world_state_get_set() {
        let mut state = WorldState::<TestKey>::new();
        assert!(!state.get(TestKey::A));
        state.set(TestKey::A, true);
        assert!(state.get(TestKey::A));
        assert!(!state.get(TestKey::B));
    }

    #[test]
    fn precondition_satisfies() {
        let mut state = WorldState::<TestKey>::new();
        state.set(TestKey::A, true);
        state.set(TestKey::B, false);

        let pre = Precondition::new()
            .requires(TestKey::A, true)
            .requires(TestKey::B, false);

        assert!(state.satisfies(&pre));

        state.set(TestKey::A, false);
        assert!(!state.satisfies(&pre));
    }

    #[test]
    fn effect_apply() {
        let state = WorldState::<TestKey>::new();
        let eff = Effect::new().sets(TestKey::A, true).sets(TestKey::B, true);
        let after = state.apply(&eff);
        assert!(after.get(TestKey::A));
        assert!(after.get(TestKey::B));
        assert!(!after.get(TestKey::C));
    }

    #[test]
    fn debug() {
        let mut state = WorldState::<TestKey>::new();
        state.set(TestKey::A, true);
        state.set(TestKey::B, false);
        state.set(TestKey::C, true);

        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("A=true"));
        assert!(debug_str.contains("B=false"));
        assert!(debug_str.contains("C=true"));
    }
}
