use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, RealField, Vector2,
    U1,
};

use crate::{
    adaptive_tessellation_processor::AdaptiveTessellationOptions, prelude::NurbsSurface, Float,
    SurfacePoint,
};

pub struct AdaptiveTessellationNode<T: RealField, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    pub(crate) id: usize,
    pub(crate) children: Vec<usize>,
    pub(crate) corners: [SurfacePoint<T, DimNameDiff<D, U1>>; 4],
    pub(crate) neighbors: [Option<usize>; 4],
    pub(crate) mid_points: [Option<SurfacePoint<T, DimNameDiff<D, U1>>>; 4],
    pub(crate) split_vertical: bool,
    pub(crate) split_horizontal: bool,
    pub(crate) horizontal: bool,
    pub(crate) u05: T,
    pub(crate) v05: T,
}

impl<T: Float, D: DimName> AdaptiveTessellationNode<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
            split_horizontal: false,
            split_vertical: false,
            horizontal: false,
            u05: T::zero(),
            v05: T::zero(),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn center(&self, surface: &NurbsSurface<T, D>) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        self.evaluate_surface(surface, self.u05, self.v05)
    }

    pub fn evaluate_corners(&mut self, surface: &NurbsSurface<T, D>) {
        //eval the center
        let inv = T::from_f64(0.5).unwrap();
        self.u05 = (self.corners[0].uv[0] + self.corners[2].uv[0]) * inv;
        self.v05 = (self.corners[0].uv[1] + self.corners[2].uv[1]) * inv;

        //eval all of the corners
        for i in 0..4 {
            let c = &self.corners[i];
            let evaled = self.evaluate_surface(surface, c.uv[0], c.uv[1]);
            self.corners[i] = evaled;
        }
    }

    pub fn evaluate_surface(
        &self,
        surface: &NurbsSurface<T, D>,
        u: T,
        v: T,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        let derivs = surface.rational_derivatives(u, v, 1);
        let pt = derivs[0][0].clone();
        let mut norm = derivs[0][1].cross(&derivs[1][0]);
        let degen = norm.magnitude_squared() < T::default_epsilon();
        if !degen {
            norm = norm.normalize();
        }
        SurfacePoint {
            point: pt.into(),
            normal: norm,
            uv: Vector2::new(u, v),
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
                let mut cornercopy: Vec<_> = corners
                    .into_iter()
                    .filter(|c| c.uv[idx] > orig[0].uv[idx] + e && c.uv[idx] < orig[2].uv[idx] - e)
                    .collect();
                cornercopy.reverse();
                [base_arr, cornercopy].concat()
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
                        self.mid_points[0] =
                            Some(self.evaluate_surface(surface, self.u05, self.corners[0].uv[1]));
                    }
                    1 => {
                        self.mid_points[1] =
                            Some(self.evaluate_surface(surface, self.corners[1].uv[0], self.v05));
                    }
                    2 => {
                        self.mid_points[2] =
                            Some(self.evaluate_surface(surface, self.u05, self.corners[2].uv[1]));
                    }
                    3 => {
                        self.mid_points[3] =
                            Some(self.evaluate_surface(surface, self.corners[0].uv[0], self.v05));
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

    pub fn should_divide(
        &mut self,
        surface: &NurbsSurface<T, D>,
        options: &AdaptiveTessellationOptions<T>,
        current_depth: usize,
    ) -> bool {
        if current_depth < options.min_depth {
            return true;
        }
        if current_depth >= options.max_depth {
            return false;
        }

        if self.has_bad_normals() {
            self.fix_normals();
            //don't divide any further when encountering a degenerate normal
            return false;
        }

        self.split_vertical = (self.corners[0].normal() - self.corners[1].normal()).norm_squared()
            > options.norm_tolerance
            || (self.corners[2].normal() - self.corners[3].normal()).norm_squared()
                > options.norm_tolerance;

        self.split_horizontal = (self.corners[1].normal() - self.corners[2].normal())
            .norm_squared()
            > options.norm_tolerance
            || (self.corners[3].normal() - self.corners[0].normal()).norm_squared()
                > options.norm_tolerance;

        if self.split_vertical || self.split_horizontal {
            return true;
        }

        let center = self.center(surface);

        return (center.normal() - self.corners[0].normal()).norm_squared()
            > options.norm_tolerance
            || (center.normal() - self.corners[1].normal()).norm_squared()
                > options.norm_tolerance
            || (center.normal() - self.corners[2].normal()).norm_squared()
                > options.norm_tolerance
            || (center.normal() - self.corners[3].normal()).norm_squared()
                > options.norm_tolerance;
    }
}
