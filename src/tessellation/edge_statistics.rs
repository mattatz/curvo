#![allow(unused)]
use crate::{misc::FloatingPoint, prelude::SurfaceTessellation};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};
use std::collections::HashMap;

/// Statistics of edges in a mesh
/// HashMap<(usize, usize), usize> -> ((edge index0, edge index1), count)
#[allow(unused)]
pub struct TessellationEdgeStatistics(HashMap<(usize, usize), usize>);

impl TessellationEdgeStatistics {
    /// Create a new `MeshEdgeStatistics` from a `Mesh`
    pub fn new<T: FloatingPoint, D>(tess: &SurfaceTessellation<T, D>) -> Self
    where
        D: DimName + DimNameSub<U1>,
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let mut edge_count: HashMap<(usize, usize), usize> = Default::default();

        // Count occurrences of each undirected edge
        for tri in tess.faces() {
            let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }

        Self(edge_count)
    }

    /// Get the count of edges
    pub fn count(&self) -> &HashMap<(usize, usize), usize> {
        &self.0
    }
}
