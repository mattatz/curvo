use std::collections::HashMap;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{misc::FloatingPoint, prelude::SurfaceTessellation};

/// Statistics of edges in a mesh
/// HashMap<(usize, usize), usize> -> ((edge index0, edge index1), count)
#[derive(Debug, Clone)]
pub struct TessellationEdgeStatistics(HashMap<(usize, usize), usize>);

impl TessellationEdgeStatistics {
    /// Create a new `MeshEdgeStatistics` from a `Mesh`
    pub fn new<T: FloatingPoint, D: DimName>(tess: &SurfaceTessellation<T, D>) -> Self
    where
        D: DimNameSub<U1>,
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
