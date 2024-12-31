use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    misc::FloatingPoint, prelude::NurbsSurface, surface::UVDirection,
    tessellation::adaptive_tessellation_node::AdaptiveTessellationNode,
};

use super::{
    adaptive_tessellation_node::{DividableDirection, NeighborDirection},
    adaptive_tessellation_option::AdaptiveTessellationOptions,
};

/// Processor for adaptive tessellation of a surface
pub struct AdaptiveTessellationProcessor<'a, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// The surface to tessellate
    surface: &'a NurbsSurface<T, D>,
    /// The created nodes for the tessellation
    nodes: Vec<AdaptiveTessellationNode<T, D>>,
}

impl<'a, T: FloatingPoint, D: DimName> AdaptiveTessellationProcessor<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new(
        surface: &'a NurbsSurface<T, D>,
        nodes: Vec<AdaptiveTessellationNode<T, D>>,
    ) -> Self {
        Self { surface, nodes }
    }

    pub fn nodes(&self) -> &[AdaptiveTessellationNode<T, D>] {
        &self.nodes
    }

    pub fn nodes_mut(&mut self) -> &mut [AdaptiveTessellationNode<T, D>] {
        &mut self.nodes
    }

    pub fn into_nodes(self) -> Vec<AdaptiveTessellationNode<T, D>> {
        self.nodes
    }

    pub fn divide(&mut self, id: usize, options: &AdaptiveTessellationOptions<T>) {
        let direction = if self.surface.u_degree() > 1 {
            UVDirection::U
        } else {
            UVDirection::V
        };
        self.iterate(id, options, 0, direction);
    }

    /// iterate over the nodes and divide them if necessary
    fn iterate(
        &mut self,
        id: usize,
        options: &AdaptiveTessellationOptions<T>,
        current_depth: usize,
        direction: UVDirection,
    ) {
        let id0 = self.nodes.len();
        let id1 = id0 + 1;

        let node = self.nodes.get_mut(id).unwrap();
        node.evaluate_corners(self.surface);

        let dividable = node.should_divide(self.surface, options, current_depth);

        node.direction = match dividable {
            Some(DividableDirection::Both) => direction,
            Some(DividableDirection::U) => UVDirection::U,
            Some(DividableDirection::V) => UVDirection::V,
            None => {
                return;
            }
        };

        let (c0, c1) = {
            match node.direction {
                UVDirection::U => {
                    let bottom = [
                        node.corners[0].clone(),
                        node.corners[1].clone(),
                        node.evaluate_mid_point(self.surface, NeighborDirection::East),
                        node.evaluate_mid_point(self.surface, NeighborDirection::West),
                    ];
                    let top = [
                        node.evaluate_mid_point(self.surface, NeighborDirection::West),
                        node.evaluate_mid_point(self.surface, NeighborDirection::East),
                        node.corners[2].clone(),
                        node.corners[3].clone(),
                    ];

                    node.assign_children([id0, id1]);

                    //assign neighbors to bottom node
                    let bottom_neighbors = [
                        node.neighbors[0],
                        node.neighbors[1],
                        Some(id1),
                        node.neighbors[3],
                    ];

                    //assign neighbors to top node
                    let top_neighbors = [
                        Some(id0),
                        node.neighbors[1],
                        node.neighbors[2],
                        node.neighbors[3],
                    ];

                    (
                        AdaptiveTessellationNode::new(id0, bottom, Some(bottom_neighbors)),
                        AdaptiveTessellationNode::new(id1, top, Some(top_neighbors)),
                    )
                }
                UVDirection::V => {
                    let left = [
                        node.corners[0].clone(),
                        node.evaluate_mid_point(self.surface, NeighborDirection::South),
                        node.evaluate_mid_point(self.surface, NeighborDirection::North),
                        node.corners[3].clone(),
                    ];
                    let right = [
                        node.evaluate_mid_point(self.surface, NeighborDirection::South),
                        node.corners[1].clone(),
                        node.corners[2].clone(),
                        node.evaluate_mid_point(self.surface, NeighborDirection::North),
                    ];

                    node.assign_children([id0, id1]);

                    let left_neighbors = [
                        node.neighbors[0],
                        Some(id1),
                        node.neighbors[2],
                        node.neighbors[3],
                    ];
                    let right_neighbors = [
                        node.neighbors[0],
                        node.neighbors[1],
                        node.neighbors[2],
                        Some(id0),
                    ];

                    (
                        AdaptiveTessellationNode::new(id0, left, Some(left_neighbors)),
                        AdaptiveTessellationNode::new(id1, right, Some(right_neighbors)),
                    )
                }
            }
        };

        self.nodes.push(c0);
        self.nodes.push(c1);

        //divide all children recursively
        for child in [id0, id1] {
            self.iterate(child, options, current_depth + 1, direction.opposite());
        }
    }
}
