pub mod constrained_triangulation;
pub mod trimmed_surface_constraints;
pub mod trimmed_surface_ext;

use std::collections::HashMap;

use super::adaptive_tessellation_node::AdaptiveTessellationNode;
use super::adaptive_tessellation_option::AdaptiveTessellationOptions;
use super::surface_tessellation::SurfaceTessellation;
use super::{ConstrainedTessellation, DividableDirection, Tessellation};
use itertools::Itertools;
use nalgebra::{ComplexField, Point2, Point3, Vector3};
use nalgebra::{Vector2, U4};
use spade::{ConstrainedDelaunayTriangulation, HasPosition, SpadeNum, Triangulation};

use crate::curve::NurbsCurve2D;
use crate::misc::FloatingPoint;
use crate::misc::PolygonBoundary;
use crate::prelude::{Contains, SurfaceTessellation3D};
use crate::region::CompoundCurve2D;
use crate::surface::{NurbsSurface3D, TrimmedSurface};
use crate::tessellation::trimmed_surface::trimmed_surface_ext::TrimmedSurfaceExt;
pub use constrained_triangulation::TrimmedSurfaceConstrainedTriangulation;
pub use trimmed_surface_constraints::*;

#[derive(Debug, Clone, Copy)]
pub struct Vertex<T: FloatingPoint> {
    point: Point3<T>,
    normal: Vector3<T>,
    uv: Vector2<T>,
}

impl<T: FloatingPoint> Vertex<T> {
    pub fn new(point: Point3<T>, normal: Vector3<T>, uv: Vector2<T>) -> Self {
        Self { point, normal, uv }
    }

    pub fn point(&self) -> Point3<T> {
        self.point
    }

    pub fn normal(&self) -> Vector3<T> {
        self.normal
    }

    pub fn uv(&self) -> Vector2<T> {
        self.uv
    }
}

impl<T: FloatingPoint + SpadeNum> HasPosition for Vertex<T> {
    type Scalar = T;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        spade::Point2::from([self.uv.x, self.uv.y])
    }
}

impl<T: FloatingPoint + SpadeNum, F> Tessellation<Option<AdaptiveTessellationOptions<T, U4, F>>>
    for TrimmedSurface<T>
where
    F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
{
    type Output = anyhow::Result<SurfaceTessellation3D<T>>;

    /// Tessellate a trimmed surface using an adaptive algorithm
    fn tessellate(&self, options: Option<AdaptiveTessellationOptions<T, U4, F>>) -> Self::Output {
        trimmed_surface_adaptive_tessellate(self, None, options)
    }
}

impl<T: FloatingPoint + SpadeNum, F>
    ConstrainedTessellation<Option<AdaptiveTessellationOptions<T, U4, F>>> for TrimmedSurface<T>
where
    F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
{
    type Constraint = TrimmedSurfaceConstraints<T>;
    type Output = anyhow::Result<SurfaceTessellation3D<T>>;

    /// Tessellate a trimmed surface using an adaptive algorithm with constraints
    fn constrained_tessellate(
        &self,
        constraints: Self::Constraint,
        adaptive_options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> Self::Output {
        trimmed_surface_adaptive_tessellate(self, Some(constraints), adaptive_options)
    }
}

/// Tessellate a trimmed surface using an adaptive algorithm with or without constraints
fn trimmed_surface_adaptive_tessellate<T: FloatingPoint + SpadeNum, S: TrimmedSurfaceExt<T, F>, F>(
    s: &S,
    constraints: Option<TrimmedSurfaceConstraints<T>>,
    options: Option<AdaptiveTessellationOptions<T, U4, F>>,
) -> anyhow::Result<SurfaceTessellation3D<T>>
where
    F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
{
    let TrimmedSurfaceConstrainedTriangulation {
        cdt,
        exterior,
        interiors,
    } = TrimmedSurfaceConstrainedTriangulation::try_new(s, constraints, options)?;

    let vmap: HashMap<_, _> = cdt
        .vertices()
        .enumerate()
        .map(|(i, v)| (v.fix(), i))
        .collect();

    let uv_exterior_boundary =
        exterior.map(|c| PolygonBoundary::new(c.iter().map(|v| v.uv.into()).collect()));
    let uv_interior_boundaries = interiors
        .into_iter()
        .map(|c| PolygonBoundary::new(c.iter().map(|v| v.uv.into()).collect()))
        .collect_vec();

    let inv_3 = T::from_f64(1. / 3.).unwrap();

    let faces = cdt
        .inner_faces()
        .filter_map(|f| {
            let vs = f.vertices();
            let tri = vs.iter().map(|v| v.as_ref()).map(|p| p.uv).collect_vec();

            let (a, b) = (tri[1] - tri[0], tri[2] - tri[1]);
            let area = a.x * b.y - a.y * b.x;
            if ComplexField::abs(area) < T::default_epsilon() {
                return None;
            }

            let center: Point2<T> = ((tri[0] + tri[1] + tri[2]) * inv_3).into();
            if uv_exterior_boundary
                .as_ref()
                .map(|exterior| exterior.contains(&center, ()).unwrap_or(false))
                .unwrap_or(true)
                && (uv_interior_boundaries.is_empty()
                    || !uv_interior_boundaries
                        .iter()
                        .any(|interior| interior.contains(&center, ()).unwrap_or(false)))
            {
                let a = vmap[&vs[0].fix()];
                let b = vmap[&vs[1].fix()];
                let c = vmap[&vs[2].fix()];
                Some([a, b, c])
            } else {
                None
            }
        })
        .collect_vec();

    // filter out isolated vertices
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut vertices = vec![];
    let vs = cdt.vertices().collect_vec();
    let remapped_faces = faces
        .iter()
        .filter_map(|face| {
            face.iter()
                .map(|v| {
                    *remap.entry(*v).or_insert_with(|| {
                        let i = vertices.len();
                        vertices.push(*vs[*v].as_ref());
                        i
                    })
                })
                .collect_array::<3>()
        })
        .collect_vec();

    let mut points = vec![];
    let mut normals = vec![];
    let mut uvs = vec![];
    vertices.iter().for_each(|v| {
        points.push(v.point);
        normals.push(v.normal);
        uvs.push(v.uv);
    });

    Ok(SurfaceTessellation {
        faces: remapped_faces,
        points,
        normals,
        uvs,
    })
}

/// Tessellate the compound curve using an adaptive algorithm recursively
fn tessellate_uv_compound_curve_adaptive<T: FloatingPoint, S: TrimmedSurfaceExt<T, F>, F>(
    curve: &CompoundCurve2D<T>,
    surface: &S,
    tolerance: T,
) -> Vec<Vertex<T>> {
    curve
        .spans()
        .iter()
        .enumerate()
        .flat_map(|(i, span)| {
            let mut vertices = tessellate_uv_curve_adaptive(span, surface, tolerance);
            if i > 0 {
                vertices.remove(0); // Skip the first vertex for spans after the first
            }
            vertices
        })
        .collect_vec()
}

/// Tessellate the curve using an adaptive algorithm recursively
fn tessellate_uv_curve_adaptive<T: FloatingPoint, S: TrimmedSurfaceExt<T, F>, F>(
    curve: &NurbsCurve2D<T>,
    surface: &S,
    tolerance: T,
) -> Vec<Vertex<T>> {
    let degree = curve.degree();
    match degree {
        1 => {
            // if the curve is a linear curve, should start tessellation from the knots
            let knots = curve.knots();
            let n = knots.len();
            let (min, max) = curve.knots_domain();
            let min = min + T::default_epsilon();
            let max = max - T::default_epsilon();

            let pts = (1..n - 2)
                .filter_map(|i| {
                    let start = knots[i].clamp(min, max);
                    let end = knots[i + 1].clamp(min, max);
                    if start == end {
                        return None;
                    }
                    let evaluated =
                        iterate_uv_curve_tessellation(curve, surface, start, end, tolerance);
                    #[allow(clippy::iter_skip_zero)]
                    if i == 1 {
                        Some(evaluated.into_iter().skip(0))
                    } else {
                        Some(evaluated.into_iter().skip(1))
                    }
                })
                .flatten();

            pts.map(|uv| {
                let p = surface.point_at(uv.x, uv.y);
                let n = surface.normal_at(uv.x, uv.y);
                Vertex::new(p, n, uv.coords)
            })
            .collect_vec()
        }
        _ => {
            let (start, end) = curve.knots_domain();
            let pts = iterate_uv_curve_tessellation(curve, surface, start, end, tolerance);
            pts.into_iter()
                .map(|uv| {
                    let p = surface.point_at(uv.x, uv.y);
                    let n = surface.normal_at(uv.x, uv.y);
                    Vertex::new(p, n, uv.coords)
                })
                .collect_vec()
        }
    }
}

fn iterate_uv_curve_tessellation<T: FloatingPoint, S: TrimmedSurfaceExt<T, F>, F>(
    curve: &NurbsCurve2D<T>,
    surface: &S,
    start: T,
    end: T,
    normal_tolerance: T,
) -> Vec<Point2<T>> {
    let (u_domain, v_domain) = surface.knots_domain();
    let eps = T::from_f64(1e-2).unwrap();
    let min = Point2::new(u_domain.0 + eps, v_domain.0 + eps);
    let max = Point2::new(u_domain.1 - eps, v_domain.1 - eps);

    let (p1, n1) = curve.point_tangent_at(start);
    let delta = end - start;
    if delta < T::from_f64(1e-8).unwrap() {
        return vec![Point2::new(
            p1.x.clamp(min.x, max.x),
            p1.y.clamp(min.y, max.y),
        )];
    }

    let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
    let (p2, n2) = curve.point_tangent_at(exact_mid);
    let (p3, n3) = curve.point_tangent_at(end);

    let flag = {
        if curve.degree() == 1 {
            // if the curve is a linear curve, we don't need to tessellate it by normal
            false
        } else {
            let diff = n2 - n1;
            let diff2 = n3 - n2;
            (diff - diff2).norm() > normal_tolerance
        }
    } || {
        let sn1 = surface.normal_at(p1.x, p1.y);
        let sn2 = surface.normal_at(p2.x, p2.y);
        let sn3 = surface.normal_at(p3.x, p3.y);
        let diff = sn2 - sn1;
        let diff2 = sn3 - sn2;
        (diff - diff2).norm() > normal_tolerance
    };
    if flag {
        let mut left_pts =
            iterate_uv_curve_tessellation(curve, surface, start, exact_mid, normal_tolerance);
        let right_pts =
            iterate_uv_curve_tessellation(curve, surface, exact_mid, end, normal_tolerance);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![
            Point2::new(p1.x.clamp(min.x, max.x), p1.y.clamp(min.y, max.y)),
            Point2::new(p3.x.clamp(min.x, max.x), p3.y.clamp(min.y, max.y)),
        ]
    }
}
