use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, RealField, Vector2,
    U1,
};

use crate::{
    misc::FloatingPoint,
    prelude::{AdaptiveTessellationOptions, NurbsSurface},
};

use super::{constraint::Constraint, SurfacePoint};

/// Node for adaptive tessellation of a surface
pub struct AdaptiveTessellationNode<T: RealField, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub(crate) id: usize,
    pub(crate) children: Vec<usize>,
    pub(crate) corners: [SurfacePoint<T, DimNameDiff<D, U1>>; 4],
    pub(crate) neighbors: [Option<usize>; 4], // [south, east, north, west] order (east & west are u direction, north & south are v direction)
    pub(crate) mid_points: [Option<SurfacePoint<T, DimNameDiff<D, U1>>>; 4],
    pub(crate) horizontal: bool,
    pub(crate) center: Vector2<T>,
    pub(crate) constraint: Option<Constraint>,
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
        Self {
            id,
            corners,
            neighbors: neighbors.unwrap_or([None, None, None, None]),
            children: vec![],
            mid_points: [None, None, None, None],
            horizontal: false,
            center: Vector2::zeros(),
            constraint: None,
        }
    }

    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = Some(constraint);
        self
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn center(&self, surface: &NurbsSurface<T, D>) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        self.evaluate_surface(surface, self.center)
    }

    pub fn evaluate_corners(&mut self, surface: &NurbsSurface<T, D>) {
        //eval the center
        let inv = T::from_f64(0.5).unwrap();
        self.center = (self.corners[0].uv + self.corners[2].uv) * inv;

        //eval all of the corners
        for i in 0..4 {
            let c = &self.corners[i];
            let evaled = self.evaluate_surface(surface, c.uv);
            self.corners[i] = evaled;
        }
    }

    pub fn evaluate_surface(
        &self,
        surface: &NurbsSurface<T, D>,
        uv: Vector2<T>,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        let derivs = surface.rational_derivatives(uv.x, uv.y, 1);
        let pt = derivs[0][0].clone();
        let mut norm = derivs[1][0].cross(&derivs[0][1]);
        let degen = norm.magnitude_squared() < T::default_epsilon();
        if !degen {
            norm = norm.normalize();
        }
        SurfacePoint {
            point: pt.into(),
            normal: norm,
            uv,
            is_normal_degenerated: degen,
        }
    }

    pub fn get_edge_corners(
        &self,
        nodes: &Vec<Self>,
        edge_index: usize,
    ) -> Vec<SurfacePoint<T, DimNameDiff<D, U1>>> {
        //if its a leaf, there are no children to obtain uvs from
        if self.is_leaf() {
            return vec![self.corners[edge_index].clone()];
        }

        if self.horizontal {
            match edge_index {
                0 => nodes[self.children[0]].get_edge_corners(nodes, 0),
                1 => [
                    nodes[self.children[0]].get_edge_corners(nodes, 1),
                    nodes[self.children[1]].get_edge_corners(nodes, 1),
                ]
                .concat(),
                2 => nodes[self.children[1]].get_edge_corners(nodes, 2),
                3 => [
                    nodes[self.children[1]].get_edge_corners(nodes, 3),
                    nodes[self.children[0]].get_edge_corners(nodes, 3),
                ]
                .concat(),
                _ => vec![],
            }
        } else {
            //vertical case
            match edge_index {
                0 => [
                    nodes[self.children[0]].get_edge_corners(nodes, 0),
                    nodes[self.children[1]].get_edge_corners(nodes, 0),
                ]
                .concat(),
                1 => nodes[self.children[1]].get_edge_corners(nodes, 1),
                2 => [
                    nodes[self.children[1]].get_edge_corners(nodes, 2),
                    nodes[self.children[0]].get_edge_corners(nodes, 2),
                ]
                .concat(),
                3 => nodes[self.children[0]].get_edge_corners(nodes, 3),
                _ => vec![],
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
                let mut corner: Vec<_> = corners
                    .into_iter()
                    .filter(|c| c.uv[idx] > orig[0].uv[idx] + e && c.uv[idx] < orig[2].uv[idx] - e)
                    .collect();
                corner.reverse();
                [base_arr, corner].concat()
            }
        }
    }

    pub fn evaluate_mid_point(
        &mut self,
        surface: &NurbsSurface<T, D>,
        index: usize,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        match self.mid_points[index] {
            Some(ref p) => p.clone(),
            None => {
                match index {
                    0 => {
                        self.mid_points[0] = Some(self.evaluate_surface(
                            surface,
                            Vector2::new(self.center.x, self.corners[0].uv.y),
                        ));
                    }
                    1 => {
                        self.mid_points[1] = Some(self.evaluate_surface(
                            surface,
                            Vector2::new(self.corners[1].uv.x, self.center.y),
                        ));
                    }
                    2 => {
                        self.mid_points[2] = Some(self.evaluate_surface(
                            surface,
                            Vector2::new(self.center.x, self.corners[2].uv.y),
                        ));
                    }
                    3 => {
                        self.mid_points[3] = Some(self.evaluate_surface(
                            surface,
                            Vector2::new(self.corners[0].uv.x, self.center.y),
                        ));
                    }
                    _ => {}
                };
                self.mid_points[index].clone().unwrap()
            }
        }
    }

    pub fn has_bad_normals(&self) -> bool {
        self.corners[0].is_normal_degenerated
            || self.corners[1].is_normal_degenerated
            || self.corners[2].is_normal_degenerated
            || self.corners[3].is_normal_degenerated
    }

    pub fn fix_normals(&mut self) {
        let l = self.corners.len();

        for i in 0..l {
            if self.corners[i].is_normal_degenerated {
                //get neighbors
                let v1 = &self.corners[(i + 1) % l];
                let v2 = &self.corners[(i + 3) % l];
                //correct the normal
                self.corners[i].normal = if v1.is_normal_degenerated {
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
    ) -> DividableDirection {
        if current_depth < options.min_depth {
            return DividableDirection::Both;
        }

        if current_depth >= options.max_depth {
            return DividableDirection::None;
        }

        if self.has_bad_normals() {
            self.fix_normals();
            //don't divide any further when encountering a degenerate normal
            return DividableDirection::None;
        }

        // println!("{}, {}", surface.v_degree() >= 2, surface.u_degree() >= 2);

        let vertical = (self.corners[0].normal() - self.corners[1].normal()).norm_squared()
            > options.norm_tolerance
            || (self.corners[2].normal() - self.corners[3].normal()).norm_squared()
                > options.norm_tolerance;
        let vertical = vertical && !matches!(self.constraint, Some(Constraint::Vertical));

        let horizontal = (self.corners[1].normal() - self.corners[2].normal()).norm_squared()
            > options.norm_tolerance
            || (self.corners[3].normal() - self.corners[0].normal()).norm_squared()
                > options.norm_tolerance;
        let horizontal = horizontal && !matches!(self.constraint, Some(Constraint::Horizontal));

        match (vertical, horizontal) {
            (true, true) => DividableDirection::Both,
            (true, false) => DividableDirection::Vertical,
            (false, true) => DividableDirection::Horizontal,
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
                    DividableDirection::Both
                } else {
                    DividableDirection::None
                }
            }
        }
    }
}

/// Enum to represent the direction in which a node can be divided
#[derive(Debug)]
pub enum DividableDirection {
    Both,
    Vertical,
    Horizontal,
    None,
}
