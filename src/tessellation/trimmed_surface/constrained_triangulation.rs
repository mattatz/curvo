use super::DividableDirection;
use itertools::Itertools;
use nalgebra::U4;
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

        let curve_tessellation_option = o
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
                        None => tessellate_uv_compound_curve_adaptive(
                            curve,
                            surface,
                            curve_tessellation_option,
                        ),
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
                        None => tessellate_uv_compound_curve_adaptive(
                            curve,
                            surface,
                            curve_tessellation_option,
                        ),
                    })
                    .collect_vec();
                (exterior, interiors)
            }
            None => {
                let exterior = surface.exterior().map(|curve| {
                    tessellate_uv_compound_curve_adaptive(curve, surface, curve_tessellation_option)
                });

                let interiors = surface
                    .interiors()
                    .iter()
                    .map(|curve| {
                        tessellate_uv_compound_curve_adaptive(
                            curve,
                            surface,
                            curve_tessellation_option,
                        )
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

        surface_division.iter().for_each(|v| {
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
