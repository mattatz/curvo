use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, RealField, Vector2,
    U1,
};

use crate::{
    misc::FloatingPoint,
    prelude::{AdaptiveTessellationOptions, NurbsSurface},
    surface::UVDirection,
};

use super::SurfacePoint;

/// Node for adaptive tessellation of a surface
pub struct AdaptiveTessellationNode<T: RealField, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    id: usize,
    children: Option<[usize; 2]>,
    pub(crate) corners: [SurfacePoint<T, DimNameDiff<D, U1>>; 4], // [left-bottom, right-bottom, right-top, left-top] order
    pub(crate) neighbors: [Option<usize>; 4], // [south, east, north, west] order (east & west are u direction, north & south are v direction)
    mid_points: [Option<SurfacePoint<T, DimNameDiff<D, U1>>>; 4], // [south, east, north, west] order
    pub(crate) direction: UVDirection,
    uv_center: Vector2<T>,
    constraint: Option<UVDirection>,
}

impl<T: FloatingPoint, D: DimName> AdaptiveTessellationNode<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new(
        id: usize,
        corners: [SurfacePoint<T, DimNameDiff<D, U1>>; 4],
        neighbors: Option<[Option<usize>; 4]>,
    ) -> Self {
        let uv_center = (corners[0].uv + corners[2].uv) * T::from_f64(0.5).unwrap();
        Self {
            id,
            corners,
            neighbors: neighbors.unwrap_or([None; 4]),
            children: None,
            mid_points: [None, None, None, None],
            direction: UVDirection::V,
            uv_center,
            constraint: None,
        }
    }

    pub fn with_constraint(mut self, constraint: UVDirection) -> Self {
        self.constraint = Some(constraint);
        self
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    pub fn assign_children(&mut self, children: [usize; 2]) {
        self.children = Some(children);
    }

    /// Evaluate the center of the node
    pub fn center(&self, surface: &NurbsSurface<T, D>) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        evaluate_surface(surface, self.uv_center)
    }

    pub fn evaluate_corners(&mut self, surface: &NurbsSurface<T, D>) {
        //eval all of the corners
        self.corners.iter_mut().for_each(|pt| {
            let e = evaluate_surface(surface, pt.uv);
            *pt = e;
        });
    }

    fn get_edge_corners(
        &self,
        nodes: &Vec<Self>,
        edge_index: usize,
    ) -> Vec<SurfacePoint<T, DimNameDiff<D, U1>>> {
        match &self.children {
            Some(children) => match self.direction {
                UVDirection::U => match edge_index {
                    0 => nodes[children[0]].get_edge_corners(nodes, 0),
                    1 => [
                        nodes[children[0]].get_edge_corners(nodes, 1),
                        nodes[children[1]].get_edge_corners(nodes, 1),
                    ]
                    .concat(),
                    2 => nodes[children[1]].get_edge_corners(nodes, 2),
                    3 => [
                        nodes[children[1]].get_edge_corners(nodes, 3),
                        nodes[children[0]].get_edge_corners(nodes, 3),
                    ]
                    .concat(),
                    _ => vec![],
                },
                UVDirection::V => match edge_index {
                    0 => [
                        nodes[children[0]].get_edge_corners(nodes, 0),
                        nodes[children[1]].get_edge_corners(nodes, 0),
                    ]
                    .concat(),
                    1 => nodes[children[1]].get_edge_corners(nodes, 1),
                    2 => [
                        nodes[children[1]].get_edge_corners(nodes, 2),
                        nodes[children[0]].get_edge_corners(nodes, 2),
                    ]
                    .concat(),
                    3 => nodes[children[0]].get_edge_corners(nodes, 3),
                    _ => vec![],
                },
            },
            None => {
                //if its a leaf, there are no children to obtain uvs from
                vec![self.corners[edge_index].clone()]
            }
        }
    }

    pub fn get_all_corners(
        &self,
        nodes: &Vec<Self>,
        edge_index: usize,
    ) -> Vec<SurfacePoint<T, DimNameDiff<D, U1>>> {
        let base_arr = vec![self.corners[edge_index].clone()];

        match self.neighbors[edge_index] {
            None => base_arr,
            Some(neighbor) => {
                let orig = &self.corners;
                //get opposite edges uvs
                let corners = nodes[neighbor].get_edge_corners(nodes, (edge_index + 2) % 4);
                let e = T::default_epsilon();

                //clip the range of uvs to match self one
                let idx = edge_index % 2;
                let corner = corners
                    .into_iter()
                    .filter(|c| c.uv[idx] > orig[0].uv[idx] + e && c.uv[idx] < orig[2].uv[idx] - e)
                    .rev()
                    .collect_vec();
                [base_arr, corner].concat()
            }
        }
    }

    pub fn evaluate_mid_point(
        &mut self,
        surface: &NurbsSurface<T, D>,
        direction: NeighborDirection,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        let index = direction as usize;
        match self.mid_points[index] {
            Some(ref p) => p.clone(),
            None => {
                let uv = match direction {
                    NeighborDirection::South => {
                        Vector2::new(self.uv_center.x, self.corners[0].uv.y)
                    }
                    NeighborDirection::East => Vector2::new(self.corners[1].uv.x, self.uv_center.y),
                    NeighborDirection::North => {
                        Vector2::new(self.uv_center.x, self.corners[2].uv.y)
                    }
                    NeighborDirection::West => Vector2::new(self.corners[0].uv.x, self.uv_center.y),
                };
                let pt = evaluate_surface(surface, uv);
                self.mid_points[index] = Some(pt.clone());
                pt
            }
        }
    }

    /// Check if the node has bad normals
    fn has_bad_normals(&self) -> bool {
        self.corners.iter().any(|c| c.is_normal_degenerated())
    }

    fn fix_normals(&mut self) {
        let l = self.corners.len();

        for i in 0..l {
            if self.corners[i].is_normal_degenerated() {
                //get neighbors
                let v1 = &self.corners[(i + 1) % l];
                let v2 = &self.corners[(i + 3) % l];
                //correct the normal
                self.corners[i].normal = if v1.is_normal_degenerated() {
                    v2.normal.clone()
                } else {
                    v1.normal.clone()
                };
            }
        }
    }

    /// Check if the node should be divided
    pub fn should_divide(
        &mut self,
        surface: &NurbsSurface<T, D>,
        options: &AdaptiveTessellationOptions<T>,
        current_depth: usize,
    ) -> Option<DividableDirection> {
        if current_depth < options.min_depth {
            return Some(DividableDirection::Both);
        }

        if current_depth >= options.max_depth {
            return None;
        }

        if self.has_bad_normals() {
            self.fix_normals();
            //don't divide any further when encountering a degenerate normal
            return None;
        }

        // println!("{}, {}", surface.v_degree() >= 2, surface.u_degree() >= 2);

        let v_direction = (self.corners[0].normal() - self.corners[1].normal()).norm_squared()
            > options.norm_tolerance
            || (self.corners[2].normal() - self.corners[3].normal()).norm_squared()
                > options.norm_tolerance;
        let v_direction = v_direction && !matches!(self.constraint, Some(UVDirection::V));

        let u_direction = (self.corners[1].normal() - self.corners[2].normal()).norm_squared()
            > options.norm_tolerance
            || (self.corners[3].normal() - self.corners[0].normal()).norm_squared()
                > options.norm_tolerance;
        let u_direction = u_direction && !matches!(self.constraint, Some(UVDirection::U));

        match (u_direction, v_direction) {
            (true, true) => Some(DividableDirection::Both),
            (true, false) => Some(DividableDirection::U),
            (false, true) => Some(DividableDirection::V),
            (false, false) => {
                let center = self.center(surface);
                if (center.normal() - self.corners[0].normal()).norm_squared()
                    > options.norm_tolerance
                    || (center.normal() - self.corners[1].normal()).norm_squared()
                        > options.norm_tolerance
                    || (center.normal() - self.corners[2].normal()).norm_squared()
                        > options.norm_tolerance
                    || (center.normal() - self.corners[3].normal()).norm_squared()
                        > options.norm_tolerance
                {
                    Some(DividableDirection::Both)
                } else {
                    None
                }
            }
        }
    }
}

/// Evaluate the surface at a given uv coordinate
fn evaluate_surface<T: FloatingPoint, D>(
    surface: &NurbsSurface<T, D>,
    uv: Vector2<T>,
) -> SurfacePoint<T, DimNameDiff<D, U1>>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let derivs = surface.rational_derivatives(uv.x, uv.y, 1);
    let pt = derivs[0][0].clone();
    let norm = derivs[1][0].cross(&derivs[0][1]);
    let is_normal_degenerated = norm.magnitude_squared() < T::default_epsilon();
    SurfacePoint::new(
        uv,
        pt.into(),
        if !is_normal_degenerated {
            norm.normalize()
        } else {
            norm
        },
        is_normal_degenerated,
    )
}

/// Enum to represent the direction in which a node can be divided
#[derive(Debug)]
pub enum DividableDirection {
    Both,
    U,
    V,
}

/// Enum to represent the direction of a neighbor
#[derive(Debug, Clone, Copy)]
pub enum NeighborDirection {
    South = 0,
    East = 1,
    North = 2,
    West = 3,
}
