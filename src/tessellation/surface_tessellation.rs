use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint,
    OVector, Vector2, Vector3, U1,
};
use simba::scalar::SupersetOf;

use crate::{
    misc::FloatingPoint, prelude::NurbsSurface,
    tessellation::adaptive_tessellation_node::AdaptiveTessellationNode,
};

use super::boundary_constraints::{BoundaryConstraints, BoundaryEvaluation};

/// Surface tessellation representation
/// This struct is used to create a mesh data from surface
#[derive(Clone, Debug)]
pub struct SurfaceTessellation<T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub(crate) points: Vec<OPoint<T, DimNameDiff<D, U1>>>,
    pub(crate) normals: Vec<OVector<T, DimNameDiff<D, U1>>>,
    pub(crate) faces: Vec<[usize; 3]>,
    pub(crate) uvs: Vec<Vector2<T>>,
}

/// 2D tessellation alias
pub type SurfaceTessellation2D<T> = SurfaceTessellation<T, Const<3>>;

/// 3D tessellation alias
pub type SurfaceTessellation3D<T> = SurfaceTessellation<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> SurfaceTessellation<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Create a new surface tessellation from surface and adaptive tessellation nodes
    pub fn new(
        surface: &NurbsSurface<T, D>,
        nodes: &Vec<AdaptiveTessellationNode<T, D>>,
        constraints: Option<BoundaryConstraints<T>>,
    ) -> Self {
        let mut tess = Self {
            points: Default::default(),
            normals: Default::default(),
            faces: Default::default(),
            uvs: Default::default(),
        };

        let boundary_evaluation = constraints.map(|c| BoundaryEvaluation::new(surface, &c));

        // Triangulate all nodes
        nodes.iter().for_each(|node| {
            tess.triangulate(surface, nodes, node, boundary_evaluation.as_ref());
        });

        tess
    }

    /// Triangulate the surface with adaptive tessellation nodes recursively
    fn triangulate(
        &mut self,
        surface: &NurbsSurface<T, D>,
        nodes: &Vec<AdaptiveTessellationNode<T, D>>,
        node: &AdaptiveTessellationNode<T, D>,
        boundary_evaluation: Option<&BoundaryEvaluation<T, D>>,
    ) {
        if node.is_leaf() {
            let corners = (0..4).map(|i| node.get_all_corners(nodes, i)).collect_vec();
            let split_id = corners
                .iter()
                .position(|c| c.len() == 2)
                .map(|i| i + 1)
                .unwrap_or(0);
            let pts = corners.into_iter().flatten().collect_vec();

            let pts = if let Some(boundary_evaluation) = boundary_evaluation {
                pts.into_iter()
                    .map(|pt| {
                        if pt.is_u_min() {
                            if let Some(closest) = boundary_evaluation.closest_point_at_u_min(&pt) {
                                return closest;
                            }
                        }
                        if pt.is_u_max() {
                            if let Some(closest) = boundary_evaluation.closest_point_at_u_max(&pt) {
                                return closest;
                            }
                        }
                        if pt.is_v_min() {
                            if let Some(closest) = boundary_evaluation.closest_point_at_v_min(&pt) {
                                return closest;
                            }
                        }
                        if pt.is_v_max() {
                            if let Some(closest) = boundary_evaluation.closest_point_at_v_max(&pt) {
                                return closest;
                            }
                        }
                        pt
                    })
                    .collect_vec()
            } else {
                pts
            };

            let base_index = self.points.len();
            let n = pts.len();
            let ids = (0..n).map(|i| base_index + i).collect_vec();
            for corner in pts.into_iter() {
                let (uv, point, normal) = corner.into_tuple();
                self.points.push(point);
                self.normals.push(normal);
                self.uvs.push(uv);
            }

            match n {
                4 => {
                    self.faces.push([ids[0], ids[1], ids[3]]);
                    self.faces.push([ids[3], ids[1], ids[2]]);
                }
                5 => {
                    let il = ids.len();
                    let a = ids[split_id];
                    let b = ids[(split_id + 1) % il];
                    let c = ids[(split_id + 2) % il];
                    let d = ids[(split_id + 3) % il];
                    let e = ids[(split_id + 4) % il];
                    self.faces.push([a, b, c]);
                    self.faces.push([a, d, e]);
                    self.faces.push([a, c, d]);
                }
                n => {
                    let center = node.center(surface);
                    self.points.push(center.point.clone());
                    self.normals.push(center.normal.clone());
                    self.uvs.push(center.uv);

                    let center_index = self.points.len() - 1;
                    let mut j = n - 1;
                    let mut i = 0;
                    while i < n {
                        self.faces.push([center_index, ids[j], ids[i]]);
                        j = i;
                        i += 1;
                    }
                }
            };
        }
    }

    /// Get the points
    pub fn points(&self) -> &Vec<OPoint<T, DimNameDiff<D, U1>>> {
        &self.points
    }

    /// Get the normals
    pub fn normals(&self) -> &Vec<OVector<T, DimNameDiff<D, U1>>> {
        &self.normals
    }

    /// Get the uvs
    pub fn uvs(&self) -> &Vec<Vector2<T>> {
        &self.uvs
    }

    /// Zip the points, normals, and uvs together
    pub fn zipped_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &OPoint<T, DimNameDiff<D, U1>>,
            &OVector<T, DimNameDiff<D, U1>>,
            &Vector2<T>,
        ),
    > {
        self.points
            .iter()
            .zip(self.normals.iter())
            .zip(self.uvs.iter())
            .map(|((p, n), uv)| (p, n, uv))
    }

    /// Zip the points, normals, and uvs together mutably
    pub fn zipped_iter_mut(
        &mut self,
    ) -> impl Iterator<
        Item = (
            &mut OPoint<T, DimNameDiff<D, U1>>,
            &mut OVector<T, DimNameDiff<D, U1>>,
            &mut Vector2<T>,
        ),
    > {
        self.points
            .iter_mut()
            .zip(self.normals.iter_mut())
            .zip(self.uvs.iter_mut())
            .map(|((p, n), uv)| (p, n, uv))
    }

    /// Get the faces
    pub fn faces(&self) -> &Vec<[usize; 3]> {
        &self.faces
    }

    /// Cast the surface tessellation to another floating point type.
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> SurfaceTessellation<F, D>
    where
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        SurfaceTessellation {
            points: self.points.iter().map(|p| p.clone().cast()).collect(),
            normals: self.normals.iter().map(|n| n.clone().cast()).collect(),
            faces: self.faces.clone(),
            uvs: self.uvs.iter().map(|uv| uv.cast()).collect(),
        }
    }
}

