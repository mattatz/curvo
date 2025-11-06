use super::{DividableDirection, Tessellation};
use itertools::Itertools;
use nalgebra::U4;
use spade::{ConstrainedDelaunayTriangulation, SpadeNum, Triangulation};

use crate::misc::FloatingPoint;
use crate::prelude::{
    AdaptiveTessellationNode, AdaptiveTessellationOptions, SurfaceTessellation,
    TrimmedSurfaceConstraints,
};
use crate::surface::TrimmedSurface;
use crate::tessellation::trimmed_surface::{tessellate_uv_compound_curve_adaptive, Vertex};

type Tri<T> = ConstrainedDelaunayTriangulation<Vertex<T>>;

pub struct ConstrainedTriangulation<T: FloatingPoint + SpadeNum> {
    pub(crate) cdt: Tri<T>,
    pub(crate) exterior: Option<Vec<Vertex<T>>>,
    pub(crate) interiors: Vec<Vec<Vertex<T>>>,
}

impl<T: FloatingPoint + SpadeNum> ConstrainedTriangulation<T> {
    pub fn try_new<F>(
        s: &TrimmedSurface<T>,
        constraints: Option<TrimmedSurfaceConstraints<T>>,
        options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> anyhow::Result<Self>
    where
        F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
    {
        let o = options.as_ref();

        let curve_tessellation_option = o
            .map(|o| o.norm_tolerance)
            .unwrap_or(T::from_f64(1e-2).unwrap());

        let (exterior, interiors) = match constraints {
            Some(constraints) => {
                anyhow::ensure!(
                    constraints.interiors().len() == s.interiors().len(),
                    "The number of interiors must match the number of trimming curves"
                );
                let exterior = s.exterior().map(|curve| match constraints.exterior() {
                    Some(constraint) => constraint
                        .iter()
                        .map(|t| {
                            let uv = curve.point_at(*t);
                            let p = s.surface().point_at(uv.x, uv.y);
                            let n = s.surface().normal_at(uv.x, uv.y);
                            Vertex::new(p, n, uv.coords)
                        })
                        .collect_vec(),
                    None => tessellate_uv_compound_curve_adaptive(
                        curve,
                        s.surface(),
                        curve_tessellation_option,
                    ),
                });
                let interiors = s
                    .interiors()
                    .iter()
                    .zip(constraints.interiors())
                    .map(|(curve, constraint)| match constraint {
                        Some(constraint) => constraint
                            .iter()
                            .map(|t| {
                                let uv = curve.point_at(*t);
                                let p = s.surface().point_at(uv.x, uv.y);
                                let n = s.surface().normal_at(uv.x, uv.y);
                                Vertex::new(p, n, uv.coords)
                            })
                            .collect_vec(),
                        None => tessellate_uv_compound_curve_adaptive(
                            curve,
                            s.surface(),
                            curve_tessellation_option,
                        ),
                    })
                    .collect_vec();
                (exterior, interiors)
            }
            None => {
                let exterior = s.exterior().map(|curve| {
                    tessellate_uv_compound_curve_adaptive(
                        curve,
                        s.surface(),
                        curve_tessellation_option,
                    )
                });

                let interiors = s
                    .interiors()
                    .iter()
                    .map(|curve| {
                        tessellate_uv_compound_curve_adaptive(
                            curve,
                            s.surface(),
                            curve_tessellation_option,
                        )
                    })
                    .collect_vec();
                (exterior, interiors)
            }
        };

        let tess = s.surface().tessellate(options);
        let SurfaceTessellation {
            points,
            normals,
            uvs,
            ..
        } = tess;

        let surface_division = points
            .into_iter()
            .zip(normals)
            .zip(uvs)
            .map(|((p, n), uv)| Vertex::new(p, n, uv))
            .collect_vec();

        let mut t = Tri::default();

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

    pub fn cdt(&self) -> &Tri<T> {
        &self.cdt
    }

    pub fn exterior(&self) -> Option<&Vec<Vertex<T>>> {
        self.exterior.as_ref()
    }

    pub fn interiors(&self) -> &Vec<Vec<Vertex<T>>> {
        &self.interiors
    }
}
