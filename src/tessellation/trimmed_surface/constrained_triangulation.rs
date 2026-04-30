use super::DividableDirection;
use itertools::Itertools;
use nalgebra::{Vector2, U4};
use spade::{ConstrainedDelaunayTriangulation, SpadeNum, Triangulation};

use crate::misc::FloatingPoint;
use crate::prelude::{
    AdaptiveTessellationNode, AdaptiveTessellationOptions, SurfaceTessellation,
    TrimmedSurfaceConstraints,
};
use crate::tessellation::trimmed_surface::trimmed_surface_ext::TrimmedSurfaceExt;
use crate::tessellation::trimmed_surface::{tessellate_uv_compound_curve_adaptive, Vertex};

type Tri<T> = ConstrainedDelaunayTriangulation<Vertex<T>>;

/// Constrained triangulation of a trimmed surface
pub struct TrimmedSurfaceConstrainedTriangulation<T: FloatingPoint + SpadeNum> {
    pub(crate) cdt: Tri<T>,
    pub(crate) exterior: Option<Vec<Vertex<T>>>,
    pub(crate) interiors: Vec<Vec<Vertex<T>>>,
}

impl<T: FloatingPoint + SpadeNum> TrimmedSurfaceConstrainedTriangulation<T> {
    pub fn try_new<S, F>(
        surface: &S,
        constraints: Option<TrimmedSurfaceConstraints<T>>,
        options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> anyhow::Result<Self>
    where
        S: TrimmedSurfaceExt<T, F>,
        F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
    {
        let o = options.as_ref();

        let angle_tolerance = o
            .map(|o| o.norm_tolerance)
            .unwrap_or(T::from_f64(1e-2).unwrap());

        let (exterior, interiors) = match constraints {
            Some(constraints) => {
                anyhow::ensure!(
                    constraints.interiors().len() == surface.interiors().len(),
                    "The number of interiors must match the number of trimming curves"
                );
                let exterior = surface
                    .exterior()
                    .map(|curve| match constraints.exterior() {
                        Some(constraint) => constraint
                            .iter()
                            .map(|t| {
                                let uv = curve.point_at(*t);
                                let p = surface.point_at(uv.x, uv.y);
                                let n = surface.normal_at(uv.x, uv.y);
                                Vertex::new(p, n, uv.coords)
                            })
                            .collect_vec(),
                        None => {
                            tessellate_uv_compound_curve_adaptive(curve, surface, angle_tolerance)
                        }
                    });
                let interiors = surface
                    .interiors()
                    .iter()
                    .zip(constraints.interiors())
                    .map(|(curve, constraint)| match constraint {
                        Some(constraint) => constraint
                            .iter()
                            .map(|t| {
                                let uv = curve.point_at(*t);
                                let p = surface.point_at(uv.x, uv.y);
                                let n = surface.normal_at(uv.x, uv.y);
                                Vertex::new(p, n, uv.coords)
                            })
                            .collect_vec(),
                        None => {
                            tessellate_uv_compound_curve_adaptive(curve, surface, angle_tolerance)
                        }
                    })
                    .collect_vec();
                (exterior, interiors)
            }
            None => {
                let exterior = surface.exterior().map(|curve| {
                    tessellate_uv_compound_curve_adaptive(curve, surface, angle_tolerance)
                });

                let interiors = surface
                    .interiors()
                    .iter()
                    .map(|curve| {
                        tessellate_uv_compound_curve_adaptive(curve, surface, angle_tolerance)
                    })
                    .collect_vec();
                (exterior, interiors)
            }
        };

        let tess = surface.tessellate_base_surface(options)?;
        let SurfaceTessellation {
            points,
            normals,
            uvs,
            ..
        } = tess;

        let mut t = Tri::default();

        let surface_division = points
            .into_iter()
            .zip(normals)
            .zip(uvs)
            .map(|((p, n), uv)| Vertex::new(p, n, uv))
            .collect_vec();

        // Collect uv-space boundary segments from the (already tessellated) constraints
        // so we can drop interior steiner points that would land on a boundary edge and
        // cause spade to split the constraint edge — that split breaks the shared-edge
        // alignment between adjacent faces.
        let boundary_segments: Vec<(Vector2<T>, Vector2<T>)> = {
            let mut segs = Vec::new();
            let push_loop = |segs: &mut Vec<(Vector2<T>, Vector2<T>)>, verts: &[Vertex<T>]| {
                if verts.len() < 2 {
                    return;
                }
                for w in verts.windows(2) {
                    segs.push((w[0].uv(), w[1].uv()));
                }
                // close the loop only if it isn't already closed
                let first = verts.first().unwrap().uv();
                let last = verts.last().unwrap().uv();
                if (first - last).norm() > T::from_f64(1e-8).unwrap() {
                    segs.push((last, first));
                }
            };
            if let Some(ext) = exterior.as_ref() {
                push_loop(&mut segs, ext);
            }
            for inter in interiors.iter() {
                push_loop(&mut segs, inter);
            }
            segs
        };

        let on_boundary_edge = |uv: Vector2<T>| -> bool {
            // Distance from uv to segment (a, b) in uv-space; reject if smaller than eps
            // AND the projection lies inside the segment.
            let eps = T::from_f64(1e-6).unwrap();
            let eps_sq = eps * eps;
            for (a, b) in boundary_segments.iter() {
                let ab = *b - *a;
                let len_sq = ab.norm_squared();
                if len_sq < T::from_f64(1e-20).unwrap() {
                    continue;
                }
                let ap = uv - *a;
                let t_param = ap.dot(&ab) / len_sq;
                if t_param < T::zero() || t_param > T::one() {
                    continue;
                }
                let proj = *a + ab * t_param;
                if (uv - proj).norm_squared() < eps_sq {
                    return true;
                }
            }
            false
        };

        surface_division.iter().for_each(|v| {
            if on_boundary_edge(v.uv()) {
                return;
            }
            let _ = t.insert(*v);
        });

        let insert_constraint = |t: &mut Tri<T>, vertices: &[Vertex<T>]| {
            let eps = T::from_f64(1e-8).unwrap();
            let skip = if let (Some(first), Some(last)) = (vertices.first(), vertices.last()) {
                // if the input vertices are closed, skip the first vertex to avoid adding a duplicate constraint
                let distance = (first.point() - last.point()).norm();
                if distance < eps {
                    1
                } else {
                    0
                }
            } else {
                0
            };

            let handles = vertices
                .iter()
                .skip(skip)
                .map(|vertex| t.insert(*vertex))
                .collect_vec();
            handles
                .into_iter()
                .circular_tuple_windows()
                .for_each(|(a, b)| {
                    if let (Ok(a), Ok(b)) = (a, b) {
                        let can_add_constraint = t.can_add_constraint(a, b);
                        if can_add_constraint {
                            t.add_constraint(a, b);
                        }
                    }
                });
        };

        if let Some(exterior) = exterior.as_ref() {
            insert_constraint(&mut t, exterior);
        }

        interiors.iter().for_each(|verts| {
            insert_constraint(&mut t, verts);
        });

        Ok(Self {
            cdt: t,
            exterior,
            interiors,
        })
    }

    #[allow(unused)]
    pub fn cdt(&self) -> &Tri<T> {
        &self.cdt
    }

    #[allow(unused)]
    pub fn exterior(&self) -> Option<&Vec<Vertex<T>>> {
        self.exterior.as_ref()
    }

    #[allow(unused)]
    pub fn interiors(&self) -> &Vec<Vec<Vertex<T>>> {
        &self.interiors
    }
}
