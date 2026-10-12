//! Values registered by name, each given a typed [`Handle`] to find it by.
//! Plain Rust, no engine: what anything that names its parts can use.

use std::marker::PhantomData;
use std::collections::HashMap;

#[derive(Debug)]
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

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self._marker == other._marker
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// By the order they were registered in.
impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

impl<T> Handle<T> {
    /// Its place in the registry's order, from 0.
    pub fn index(&self) -> usize {
        self.id
    }

    /// The handle stored as its index ([`Handle::index`]): valid in the
    /// registry that handed it out.
    pub fn from_index(id: usize) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Registry<T> {
    items: Vec<T>,
    lookup: HashMap<String, Handle<T>>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self::new()
    }
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

    /// Every value with its handle, in the order they were registered.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.items.iter().enumerate().map(|(id, item)| {
            (
                Handle {
                    id,
                    _marker: PhantomData,
                },
                item,
            )
        })
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
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
    use super::Registry;

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
