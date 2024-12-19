use std::collections::HashMap;

use super::adaptive_tessellation_option::AdaptiveTessellationOptions;
use super::surface_tessellation::SurfaceTessellation;
use super::Tessellation;
use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::Vector2;
use nalgebra::{ComplexField, Point2, Point3, Vector3};
use spade::handles::FixedVertexHandle;
use spade::{
    ConstrainedDelaunayTriangulation, HasPosition, Point2 as SP2, SpadeNum, Triangulation,
};

use crate::curve::NurbsCurve2D;
use crate::misc::FloatingPoint;
use crate::misc::PolygonBoundary;
use crate::prelude::{Contains, SurfaceTessellation3D};
use crate::surface::{NurbsSurface3D, TrimmedSurface};

#[derive(Debug, Clone, Copy)]
struct Vertex<T: FloatingPoint> {
    point: Point3<T>,
    normal: Vector3<T>,
    uv: Vector2<T>,
}

impl<T: FloatingPoint> Vertex<T> {
    pub fn new(point: Point3<T>, normal: Vector3<T>, uv: Vector2<T>) -> Self {
        Self { point, normal, uv }
    }
}

impl<T: FloatingPoint + SpadeNum> HasPosition for Vertex<T> {
    type Scalar = T;

    fn position(&self) -> spade::Point2<Self::Scalar> {
        spade::Point2::from([self.uv.x, self.uv.y])
    }
}

type Tri<T> = ConstrainedDelaunayTriangulation<Vertex<T>>;

impl<T: FloatingPoint + SpadeNum> Tessellation for TrimmedSurface<T> {
    type Option = Option<AdaptiveTessellationOptions<T>>;
    type Output = SurfaceTessellation3D<T>;

    fn tessellate(&self, options: Self::Option) -> Self::Output {
        let o = options.as_ref();

        let curve_tessellation_option = o
            .clone()
            .map(|o| o.norm_tolerance)
            .unwrap_or(T::from_f64(1e-2).unwrap());

        let exterior = match self.exterior() {
            Some(curve) => Some(tessellate_uv_curve_adaptive(
                curve,
                self.surface(),
                curve_tessellation_option,
            )),
            None => None,
        };

        let interiors = self
            .interiors()
            .iter()
            .map(|curve| {
                tessellate_uv_curve_adaptive(curve, self.surface(), curve_tessellation_option)
            })
            .collect_vec();

        let tess = self.surface().tessellate(options);
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
            let handles = vertices.iter().map(|v| t.insert(*v)).collect_vec();
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

        let mut vertices = vec![];

        let vmap: HashMap<_, _> = t
            .vertices()
            .enumerate()
            .map(|(i, v)| {
                let p = v.as_ref();
                vertices.push(p.uv);
                (v.fix(), i)
            })
            .collect();

        let uv_exterior_boundary =
            exterior.map(|c| PolygonBoundary::new(c.iter().map(|v| v.uv.into()).collect()));
        let uv_interior_boundaries = interiors
            .into_iter()
            .map(|c| PolygonBoundary::new(c.iter().map(|v| v.uv.into()).collect()))
            .collect_vec();

        let inv_3 = T::from_f64(1. / 3.).unwrap();

        let faces = t
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
                        || uv_interior_boundaries
                            .iter()
                            .any(|interior| !interior.contains(&center, ()).unwrap_or(false)))
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

        let mut points = vec![];
        let mut normals = vec![];
        let mut uvs = vec![];
        t.vertices().for_each(|v| {
            let v = v.as_ref();
            points.push(v.point);
            normals.push(v.normal);
            uvs.push(v.uv);
        });

        SurfaceTessellation {
            faces,
            points,
            normals,
            uvs,
        }
    }
}

/// Tessellate the curve using an adaptive algorithm recursively
fn tessellate_uv_curve_adaptive<T: FloatingPoint>(
    curve: &NurbsCurve2D<T>,
    surface: &NurbsSurface3D<T>,
    tolerance: T,
) -> Vec<Vertex<T>> {
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

fn iterate_uv_curve_tessellation<T: FloatingPoint>(
    curve: &NurbsCurve2D<T>,
    surface: &NurbsSurface3D<T>,
    start: T,
    end: T,
    normal_tolerance: T,
) -> Vec<Point2<T>> {
    let (p1, n1) = curve.point_tangent_at(start);
    let delta = end - start;
    if delta < T::from_f64(1e-8).unwrap() {
        return vec![p1];
    }

    let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
    let (p2, n2) = curve.point_tangent_at(exact_mid);
    let (p3, n3) = curve.point_tangent_at(end);

    let flag = {
        let diff = &n2 - &n1;
        let diff2 = &n3 - &n2;
        (diff - diff2).norm() > normal_tolerance
    } || {
        let sn1 = surface.normal_at(p1.x, p1.y);
        let sn2 = surface.normal_at(p2.x, p2.y);
        let sn3 = surface.normal_at(p3.x, p3.y);
        let diff = &sn2 - &sn1;
        let diff2 = &sn3 - &sn2;
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
        vec![p1, p3]
    }
}
