use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    misc::FloatingPoint, prelude::NurbsSurface,
    tessellation::adaptive_tessellation_node::AdaptiveTessellationNode,
};

use super::{
    adaptive_tessellation_node::DividableDirection,
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
    pub(crate) surface: &'a NurbsSurface<T, D>,
    /// The created nodes for the tessellation
    pub(crate) nodes: Vec<AdaptiveTessellationNode<T, D>>,
}

impl<'a, T: FloatingPoint, D: DimName> AdaptiveTessellationProcessor<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn north(
        &self,
        index: usize,
        i: usize,
        _j: usize,
        divs_u: usize,
        _divs_v: usize,
    ) -> Option<&AdaptiveTessellationNode<T, D>> {
        if i == 0 {
            None
        } else {
            Some(&self.nodes[index - divs_u])
        }
    }

    pub fn south(
        &self,
        index: usize,
        i: usize,
        _j: usize,
        divs_u: usize,
        divs_v: usize,
    ) -> Option<&AdaptiveTessellationNode<T, D>> {
        if i == divs_v - 1 {
            None
        } else {
            Some(&self.nodes[index + divs_u])
        }
    }

    pub fn east(
        &self,
        index: usize,
        _i: usize,
        j: usize,
        divs_u: usize,
        _divs_v: usize,
    ) -> Option<&AdaptiveTessellationNode<T, D>> {
        if j == divs_u - 1 {
            None
        } else {
            Some(&self.nodes[index + 1])
        }
    }

    pub fn west(
        &self,
        index: usize,
        _i: usize,
        j: usize,
        _divs_u: usize,
        _divs_v: usize,
    ) -> Option<&AdaptiveTessellationNode<T, D>> {
        if j == 0 {
            None
        } else {
            Some(&self.nodes[index - 1])
        }
    }

    pub fn divide(&mut self, id: usize, options: &AdaptiveTessellationOptions<T>) {
        let horizontal = self.surface.u_degree() > 1;
        self.iterate(id, options, 0, horizontal);
    }

    /// Iterate over the nodes and divide them if necessary
    /// todo: should not divide if a degree on current direction is 1 & duplicate vertices to avoid share a normal
    fn iterate(
        &mut self,
        id: usize,
        options: &AdaptiveTessellationOptions<T>,
        current_depth: usize,
        horizontal: bool,
    ) {
        let id0 = self.nodes.len();
        let id1 = id0 + 1;

        let node = self.nodes.get_mut(id).unwrap();
        node.evaluate_corners(self.surface);

        let dividable = node.should_divide(self.surface, options, current_depth);

        node.horizontal = match dividable {
            DividableDirection::Both => horizontal,
            DividableDirection::Vertical => false,
            DividableDirection::Horizontal => true,
            DividableDirection::None => {
                return;
            }
        };

        let (c0, c1) = {
            if node.horizontal {
                let bottom = [
                    node.corners[0].clone(),
                    node.corners[1].clone(),
                    node.evaluate_mid_point(self.surface, 1),
                    node.evaluate_mid_point(self.surface, 3),
                ];
                let top = [
                    node.evaluate_mid_point(self.surface, 3),
                    node.evaluate_mid_point(self.surface, 1),
                    node.corners[2].clone(),
                    node.corners[3].clone(),
                ];

                node.children = vec![id0, id1];

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
            } else {
                let left = [
                    node.corners[0].clone(),
                    node.evaluate_mid_point(self.surface, 0),
                    node.evaluate_mid_point(self.surface, 2),
                    node.corners[3].clone(),
                ];
                let right = [
                    node.evaluate_mid_point(self.surface, 0),
                    node.corners[1].clone(),
                    node.corners[2].clone(),
                    node.evaluate_mid_point(self.surface, 2),
                ];

                node.children = vec![id0, id1];

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
        };

        self.nodes.push(c0);
        self.nodes.push(c1);

        //divide all children recursively
        for child in [id0, id1] {
            self.iterate(child, options, current_depth + 1, !horizontal);
        }
    }
}
