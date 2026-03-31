use std::collections::BTreeMap;

use nalgebra::Vector2;
use ordered_float::OrderedFloat;

use crate::misc::FloatingPoint;

/// An edge in the advancing front, stored as a pair of vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrontEdge {
    pub v0: usize,
    pub v1: usize,
}

/// The advancing front: a set of edges forming the boundary between
/// the meshed region and the unmeshed interior.
pub struct Front<T: FloatingPoint> {
    /// Active edges sorted by 3D length (shortest first for greedy selection).
    /// Key = (OrderedFloat(length), edge_id) for unique ordering.
    edges: BTreeMap<(OrderedFloat<f64>, usize), FrontEdge>,
    /// Edge counter for unique IDs.
    next_id: usize,
    /// UV positions of all vertices.
    uv_positions: Vec<Vector2<T>>,
}

impl<T: FloatingPoint> Front<T> {
    pub fn new(uv_positions: Vec<Vector2<T>>) -> Self {
        Self {
            edges: BTreeMap::new(),
            next_id: 0,
            uv_positions,
        }
    }

    /// Add an edge to the front with its 3D length.
    pub fn push(&mut self, edge: FrontEdge, length_3d: T) {
        let id = self.next_id;
        self.next_id += 1;
        let key = (
            OrderedFloat(T::to_f64(&length_3d).unwrap()),
            id,
        );
        self.edges.insert(key, edge);
    }

    /// Pop the shortest edge from the front.
    pub fn pop_shortest(&mut self) -> Option<FrontEdge> {
        let key = *self.edges.keys().next()?;
        self.edges.remove(&key)
    }

    /// Check if the front is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Number of remaining edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Get the UV position of a vertex.
    pub fn uv(&self, idx: usize) -> Vector2<T> {
        self.uv_positions[idx]
    }

    /// Add a new UV position, returning its index.
    pub fn add_vertex(&mut self, uv: Vector2<T>) -> usize {
        let idx = self.uv_positions.len();
        self.uv_positions.push(uv);
        idx
    }
}
