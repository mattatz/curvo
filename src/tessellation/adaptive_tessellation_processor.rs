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

    pub fn into_nodes(self) -> Vec<AdaptiveTessellationNode<T, D>> {
        self.nodes
    }

    pub fn divide<F>(&mut self, id: usize, options: &AdaptiveTessellationOptions<T, D, F>)
    where
        F: Fn(&AdaptiveTessellationNode<T, D>) -> Option<DividableDirection> + Copy,
    {
        let direction = if self.surface.u_degree() > 1 {
            UVDirection::U
        } else {
            UVDirection::V
        };
        self.iterate(id, options, 0, direction);
    }

    /// iterate over the nodes and divide them if necessary
    fn iterate<F>(
        &mut self,
        id: usize,
        options: &AdaptiveTessellationOptions<T, D, F>,
        current_depth: usize,
        direction: UVDirection,
    ) where
        F: Fn(&AdaptiveTessellationNode<T, D>) -> Option<DividableDirection> + Copy,
    {
        let next_node_id_0 = self.nodes.len();
        let next_node_id_1 = next_node_id_0 + 1;

        let node = self.nodes.get_mut(id).unwrap();
        let dividable_direction = if current_depth < options.min_depth {
            Some(DividableDirection::Both)
        } else if current_depth >= options.max_depth {
            None
        } else {
            options
                .divider
                .and_then(|f| f(node))
                .or(node.should_divide(options.norm_tolerance))
            // node.should_divide(options.norm_tolerance).or(options.divider.and_then(|f| f(node)))
        };

        // set the divided direction of the node
        node.direction = match dividable_direction {
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
                    let east_mid = node.evaluate_mid_point(self.surface, NeighborDirection::East);
                    let west_mid = node.evaluate_mid_point(self.surface, NeighborDirection::West);

                    // counter-clockwise order [south, east, north, west]
                    let bottom = [
                        node.corners[0].clone(), // left-bottom
                        node.corners[1].clone(), // right-bottom
                        east_mid.clone(),
                        west_mid.clone(),
                    ];
                    let top = [
                        west_mid,
                        east_mid,
                        node.corners[2].clone(), // right-top
                        node.corners[3].clone(), // left-top
                    ];

                    node.assign_children([next_node_id_0, next_node_id_1]);

                    // assign neighbors to bottom node
                    let bottom_neighbors = [
                        *node.neighbors.at(NeighborDirection::South),
                        *node.neighbors.at(NeighborDirection::East),
                        Some(next_node_id_1), // top as north neighbor
                        *node.neighbors.at(NeighborDirection::West),
                    ];

                    // assign neighbors to top node
                    let top_neighbors = [
                        Some(next_node_id_0), // bottom as south neighbor
                        *node.neighbors.at(NeighborDirection::East),
                        *node.neighbors.at(NeighborDirection::North),
                        *node.neighbors.at(NeighborDirection::West),
                    ];

                    (
                        AdaptiveTessellationNode::new(next_node_id_0, bottom, bottom_neighbors),
                        AdaptiveTessellationNode::new(next_node_id_1, top, top_neighbors),
                    )
                }
                UVDirection::V => {
                    let south_mid = node.evaluate_mid_point(self.surface, NeighborDirection::South);
                    let north_mid = node.evaluate_mid_point(self.surface, NeighborDirection::North);

                    let left = [
                        node.corners[0].clone(), // left-bottom
                        south_mid.clone(),
                        north_mid.clone(),
                        node.corners[3].clone(), // left-top
                    ];
                    let right = [
                        south_mid,
                        node.corners[1].clone(), // right-bottom
                        node.corners[2].clone(), // right-top
                        north_mid,
                    ];

                    node.assign_children([next_node_id_0, next_node_id_1]);

                    let left_neighbors = [
                        *node.neighbors.at(NeighborDirection::South),
                        Some(next_node_id_1), // right as east neighbor
                        *node.neighbors.at(NeighborDirection::North),
                        *node.neighbors.at(NeighborDirection::West),
                    ];
                    let right_neighbors = [
                        *node.neighbors.at(NeighborDirection::South),
                        *node.neighbors.at(NeighborDirection::East),
                        *node.neighbors.at(NeighborDirection::North),
                        Some(next_node_id_0), // left as west neighbor
                    ];

                    (
                        AdaptiveTessellationNode::new(next_node_id_0, left, left_neighbors),
                        AdaptiveTessellationNode::new(next_node_id_1, right, right_neighbors),
                    )
                }
            }
        };

        self.nodes.push(c0);
        self.nodes.push(c1);

        // divide all children recursively
        for next_id in [next_node_id_0, next_node_id_1] {
            self.iterate(next_id, options, current_depth + 1, direction.opposite());
        }
    }
}
