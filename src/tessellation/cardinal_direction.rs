use std::ops::{Index, IndexMut};

use crate::tessellation::adaptive_tessellation_node::NeighborDirection;

/// Cardinal direction of a node
#[derive(Clone, Debug)]
pub struct CardinalDirection<T>([Option<T>; 4]); // [south, east, north, west] order

impl<T> CardinalDirection<T> {
    pub fn new(south: Option<T>, east: Option<T>, north: Option<T>, west: Option<T>) -> Self {
        Self([south, east, north, west])
    }

    /// Get the value at a given direction
    pub fn at(&self, direction: NeighborDirection) -> &Option<T> {
        &self.0[Into::<usize>::into(direction)]
    }
}

impl<T> Default for CardinalDirection<T> {
    fn default() -> Self {
        Self([None, None, None, None])
    }
}

impl<T> Index<usize> for CardinalDirection<T> {
    type Output = Option<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for CardinalDirection<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> From<[Option<T>; 4]> for CardinalDirection<T> {
    fn from(array: [Option<T>; 4]) -> Self {
        Self(array)
    }
}
