use std::marker::PhantomData;

use bevy::platform::collections::HashMap;

#[derive(Copy, Debug)]
pub struct Handle<T> {
    id: usize,
    _marker: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            _marker: self._marker.clone(),
        }
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self._marker == other._marker
    }
}

pub struct Registry<T> {
    items: Vec<T>,
    lookup: HashMap<String, Handle<T>>,
}

impl<T> Registry<T> {
    pub fn new() -> Self {
        Registry {
            items: vec![],
            lookup: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, value: T) -> Handle<T> {
        let id = self.items.len();
        let handle = Handle {
            id,
            _marker: PhantomData,
        };

        self.items.push(value);
        self.lookup.insert(name.into(), handle.clone());

        handle
    }

    pub fn get(&self, handle: &Handle<T>) -> &T {
        &self.items.get(handle.id).expect("a valid handle")
    }

    pub fn get_by_name(&self, name: impl Into<String>) -> Option<Handle<T>> {
        self.lookup.get(&name.into()).map(Clone::clone)
    }

    pub fn name_of(&self, handle: Handle<T>) -> Option<&String> {
        self.lookup
            .iter()
            .find(|x| x.1.id == handle.id)
            .map(|x| x.0)
    }
}

#[cfg(test)]
mod test {
    use crate::Registry;

    #[test]
    fn sanity() {
        let mut registry = Registry::new();
        let one = registry.register("one", 1);
        let two = registry.register("two", 2);
        let three = registry.register("three", 3);

        assert_eq!(*registry.get(&one), 1);
        assert_eq!(*registry.get(&two), 2);
        assert_eq!(*registry.get(&three), 3);

        let one_handle = registry.get_by_name("one").unwrap();
        assert_eq!(one_handle, one);
        assert_eq!(*registry.get(&one_handle), 1);
    }
}
