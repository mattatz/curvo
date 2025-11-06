use std::borrow::Cow;

use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff,
    DimNameSub, DimNameSum, OMatrix, OPoint, OVector, Point3, Point4, RealField, Vector2, Vector3,
    U1,
};
use simba::scalar::SupersetOf;

use crate::{
    curve::nurbs_curve::{dehomogenize, NurbsCurve, NurbsCurve3D},
    misc::{
        binomial::Binomial, transformable::Transformable, transpose_control_points, FloatingPoint,
        Invertible, Ray,
    },
    prelude::{
        try_interpolate_control_points, AdaptiveTessellationOptions, KnotVector,
        SurfaceTessellation, Tessellation,
    },
    SurfaceClosestParameterNewton, SurfaceClosestParameterProblem,
};

use super::{FlipDirection, UVDirection};

/// NURBS surface representation
/// by generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsSurface<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    /// control points with homogeneous coordinates
    /// the last element of the vector is the `weight`
    control_points: Vec<Vec<OPoint<T, D>>>,
    u_degree: usize,
    v_degree: usize,
    u_knots: KnotVector<T>,
    v_knots: KnotVector<T>,
}

/// 2D NURBS surface alias
pub type NurbsSurface2D<T> = NurbsSurface<T, Const<3>>;
/// 3D NURBS surface alias
pub type NurbsSurface3D<T> = NurbsSurface<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new<U: Into<KnotVector<T>>, V: Into<KnotVector<T>>>(
        u_degree: usize,
        v_degree: usize,
        u_knots: U,
        v_knots: V,
        control_points: Vec<Vec<OPoint<T, D>>>,
    ) -> Self {
        Self {
            u_degree,
            v_degree,
            u_knots: u_knots.into(),
            v_knots: v_knots.into(),
            control_points,
        }
    }

    pub fn u_degree(&self) -> usize {
        self.u_degree
    }

    pub fn v_degree(&self) -> usize {
        self.v_degree
    }

    pub fn u_knots(&self) -> &KnotVector<T> {
        &self.u_knots
    }

    pub fn v_knots(&self) -> &KnotVector<T> {
        &self.v_knots
    }

    /// Get the u domain of the knot vector by degree
    pub fn u_knots_domain(&self) -> (T, T) {
        self.u_knots.domain(self.u_degree)
    }

    /// Get the v domain of the knot vector by degree
    pub fn v_knots_domain(&self) -> (T, T) {
        self.v_knots.domain(self.v_degree)
    }

    pub fn knots_domain_at(&self, direction: UVDirection) -> (T, T) {
        match direction {
            UVDirection::U => self.u_knots_domain(),
            UVDirection::V => self.v_knots_domain(),
        }
    }

    /// Get the u and v domain of the knot vector by degree
    pub fn knots_domain(&self) -> ((T, T), (T, T)) {
        (self.u_knots_domain(), self.v_knots_domain())
    }

    /// Get the u and v domain interval of the knot vector by degree
    pub fn knots_domain_interval(&self) -> (T, T) {
        let (u, v) = self.knots_domain();
        (u.1 - u.0, v.1 - v.0)
    }

    /// Remap the u knot vector to a new domain [start, end]
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// let plane = NurbsSurface3D::<f64>::plane(Point3::origin(), Vector3::x(), Vector3::y());
    /// let (u_min, u_max) = plane.u_knots_domain();
    /// assert_eq!(u_min, 0.);
    /// assert_eq!(u_max, 1.);
    /// let mut remapped = plane.clone();
    /// remapped.remap_u_knots(10., 20.);
    /// let (u_min, u_max) = remapped.u_knots_domain();
    /// assert_eq!(u_min, 10.);
    /// assert_eq!(u_max, 20.);
    /// ```
    pub fn remap_u_knots(&mut self, start: T, end: T) {
        self.u_knots = self.u_knots.remap(self.u_degree, start, end);
    }

    /// Remap the v knot vector to a new domain [start, end]
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// let plane = NurbsSurface3D::<f64>::plane(Point3::origin(), Vector3::x(), Vector3::y());
    /// let (v_min, v_max) = plane.v_knots_domain();
    /// assert_eq!(v_min, 0.);
    /// assert_eq!(v_max, 1.);
    /// let mut remapped = plane.clone();
    /// remapped.remap_v_knots(30., 40.);
    /// let (v_min, v_max) = remapped.v_knots_domain();
    /// assert_eq!(v_min, 30.);
    /// assert_eq!(v_max, 40.);
    /// ```
    pub fn remap_v_knots(&mut self, start: T, end: T) {
        self.v_knots = self.v_knots.remap(self.v_degree, start, end);
    }

    /// Remap both u and v knot vectors to new domains
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// let plane = NurbsSurface3D::<f64>::plane(Point3::origin(), Vector3::x(), Vector3::y());
    /// let ((u_min, u_max), (v_min, v_max)) = plane.knots_domain();
    /// assert_eq!(u_min, 0.);
    /// assert_eq!(u_max, 1.);
    /// assert_eq!(v_min, 0.);
    /// assert_eq!(v_max, 1.);
    /// let mut remapped = plane.clone();
    /// remapped.remap_knots(10., 20., 30., 40.);
    /// let ((u_min, u_max), (v_min, v_max)) = remapped.knots_domain();
    /// assert_eq!(u_min, 10.);
    /// assert_eq!(u_max, 20.);
    /// assert_eq!(v_min, 30.);
    /// assert_eq!(v_max, 40.);
    /// ```
    pub fn remap_knots(&mut self, u_start: T, u_end: T, v_start: T, v_end: T) {
        self.remap_u_knots(u_start, u_end);
        self.remap_v_knots(v_start, v_end);
    }

    pub fn control_points(&self) -> &Vec<Vec<OPoint<T, D>>> {
        &self.control_points
    }

    /// Get the transposed control points
    pub fn transposed_control_points(&self) -> Vec<Vec<OPoint<T, D>>> {
        transpose_control_points(&self.control_points)
    }

    pub fn dehomogenized_control_points(&self) -> Vec<Vec<OPoint<T, DimNameDiff<D, U1>>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        self.control_points
            .iter()
            .map(|row| row.iter().map(|p| dehomogenize(p).unwrap()).collect())
            .collect()
    }

    /// Evaluate the surface at the given u, v parameters to get a point
    pub fn point_at(&self, u: T, v: T) -> OPoint<T, DimNameDiff<D, U1>> {
        let p = self.point(u, v);
        dehomogenize(&p).unwrap()
    }

    /// Evaluate the surface at the given u, v parameters to get a point
    pub fn point(&self, u: T, v: T) -> OPoint<T, D> {
        let n = self.u_knots.len() - self.u_degree - 2;
        let m = self.v_knots.len() - self.v_degree - 2;

        let knot_span_index_u = self.u_knots.find_knot_span_index(n, self.u_degree, u);
        let knot_span_index_v = self.v_knots.find_knot_span_index(m, self.v_degree, v);
        let u_basis_vals = self
            .u_knots
            .basis_functions(knot_span_index_u, u, self.u_degree);
        let v_basis_vals = self
            .v_knots
            .basis_functions(knot_span_index_v, v, self.v_degree);
        let uind = knot_span_index_u - self.u_degree;

        let mut position = OPoint::<T, D>::origin();
        for l in 0..=self.v_degree {
            let mut temp = OPoint::<T, D>::origin();
            let vind = knot_span_index_v - self.v_degree + l;

            // sample u isoline
            for k in 0..=self.u_degree {
                temp.coords += &self.control_points[uind + k][vind].coords * u_basis_vals[k];
            }

            // add point from u isoline
            position.coords += temp.coords * v_basis_vals[l];
        }

        position
    }

    // Compute a regularly spaced grid of points on surface.
    // Generally, this algorithm is faster than directly evaluating these by pre-computing all of the basis functions
    pub fn regular_sample_points(
        &self,
        divs_u: usize,
        divs_v: usize,
    ) -> Vec<Vec<OPoint<T, DimNameDiff<D, U1>>>> {
        let (knot_spans_u, bases_u) = self
            .u_knots
            .regulary_spaced_basis_functions(self.u_degree, divs_u);
        let (knot_spans_v, bases_v) = self
            .v_knots
            .regulary_spaced_basis_functions(self.v_degree, divs_v);
        let mut pts = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];

            for j in 0..=divs_v {
                let pt = self.point_given_bases_knot_spans(
                    knot_spans_u[i],
                    knot_spans_v[j],
                    &bases_u[i],
                    &bases_v[j],
                );
                row.push(dehomogenize(&pt).unwrap());
            }

            pts.push(row);
        }

        pts
    }

    /// Compute a point on the surface given the basis functions and knot spans
    fn point_given_bases_knot_spans(
        &self,
        knot_span_u: usize,
        knot_span_v: usize,
        bases_u: &[T],
        bases_v: &[T],
    ) -> OPoint<T, D> {
        let mut position = OPoint::<T, D>::origin();

        // could be precomputed
        let uind = knot_span_u - self.u_degree;
        let mut vind = knot_span_v - self.v_degree;

        for l in 0..(self.v_degree + 1) {
            let mut temp = OPoint::<T, D>::origin();

            for k in 0..(self.u_degree + 1) {
                temp.coords += &self.control_points[uind + k][vind].coords * bases_u[k];
            }

            vind += 1;

            position.coords += temp.coords * bases_v[l];
        }

        position
    }

    // Compute a regularly spaced grid of normals on surface.
    pub fn regular_sample_normals(
        &self,
        divs_u: usize,
        divs_v: usize,
    ) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> {
        let ders = self.regular_sample_rational_derivatives(divs_u, divs_v, 1);
        ders.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|der| {
                        let v0 = &der[1][0];
                        let v1 = &der[0][1];
                        v0.cross(v1).normalize()
                    })
                    .collect()
            })
            .collect()
    }

    // Compute a regularly spaced grid of rational derivatives on surface.
    #[allow(clippy::type_complexity)]
    pub fn regular_sample_rational_derivatives(
        &self,
        divs_u: usize,
        divs_v: usize,
        derivs: usize,
    ) -> Vec<Vec<Vec<Vec<OVector<T, DimNameDiff<D, U1>>>>>> {
        let ders = self.regular_sample_derivatives(divs_u, divs_v, derivs);

        let mut rat_ders = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];
            for j in 0..=divs_v {
                let skl = rational_derivatives(&ders[i][j], derivs);
                row.push(skl);
            }
            rat_ders.push(row);
        }

        rat_ders
    }

    /// Compute a regularly spaced grid of derivatives on a surface.
    #[allow(clippy::type_complexity)]
    pub fn regular_sample_derivatives(
        &self,
        divs_u: usize,
        divs_v: usize,
        derivs: usize,
    ) -> Vec<Vec<Vec<Vec<OVector<T, D>>>>> {
        let (knot_spans_u, bases_u) = self
            .u_knots
            .regularly_spaced_derivative_basis_functions(self.u_degree, divs_u);
        let (knot_spans_v, bases_v) = self
            .v_knots
            .regularly_spaced_derivative_basis_functions(self.v_degree, divs_v);
        let mut ders = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];

            for j in 0..=divs_v {
                row.push(self.derivatives_given_bases_knot_spans(
                    knot_spans_u[i],
                    knot_spans_v[j],
                    &bases_u[i],
                    &bases_v[j],
                    derivs,
                ));
            }

            ders.push(row);
        }

        ders
    }

    /// Compute the derivatives given the basis functions and knot spans
    pub fn derivatives_given_bases_knot_spans(
        &self,
        knot_span_u: usize,
        knot_span_v: usize,
        bases_u: &[Vec<T>],
        bases_v: &[Vec<T>],
        derivs: usize,
    ) -> Vec<Vec<OVector<T, D>>> {
        let du = if derivs < self.u_degree {
            derivs
        } else {
            self.u_degree
        };
        let dv = if derivs < self.v_degree {
            derivs
        } else {
            self.v_degree
        };
        let mut skl = vec![vec![OVector::<T, D>::zeros(); du + 1]; dv + 1];

        for k in 0..=du {
            let mut temp = vec![];
            for s in 0..=self.v_degree {
                let mut row = OVector::<T, D>::zeros();
                for r in 0..=self.u_degree {
                    row += &self.control_points[knot_span_u - self.u_degree + r]
                        [knot_span_v - self.v_degree + s]
                        .coords
                        * bases_u[k][r];
                }
                temp.push(row);
            }

            let nk = derivs - k;
            let dd = if nk < dv { nk } else { dv };

            for l in 0..=dd {
                let mut v = OVector::<T, D>::zeros();
                for s in 0..=self.v_degree {
                    v += &temp[s] * bases_v[l][s];
                }
                skl[k][l] = v;
            }
        }

        skl
    }

    /// Regularly tessellate the surface into a meshable form
    /// This tessellation is faster than adaptive one because of pre-computed basis functions
    /// There is trade-off between speed and mesh quality
    pub fn regular_tessellate(&self, divs_u: usize, divs_v: usize) -> SurfaceTessellation<T, D> {
        let points = self.regular_sample_points(divs_u, divs_v);
        let ders = self.regular_sample_normals(divs_u, divs_v);
        let u_span = self.u_knots.regularly_spaced_span(self.u_degree, divs_u);
        let v_span = self.v_knots.regularly_spaced_span(self.v_degree, divs_v);

        let points: Vec<_> = points.into_iter().flatten().collect();
        let normals: Vec<_> = ders.into_iter().flatten().collect();
        let faces = (0..divs_u)
            .flat_map(|iu| {
                let ioff = iu * (divs_v + 1);
                (0..divs_v).flat_map(move |iv| {
                    [
                        [ioff + iv, ioff + iv + divs_v + 2, ioff + iv + 1],
                        [ioff + iv, ioff + iv + divs_v + 1, ioff + iv + divs_v + 2],
                    ]
                })
            })
            .collect();
        let uvs = (0..=divs_u)
            .flat_map(|iu| {
                let iu = T::from_usize(iu).unwrap();
                let u = u_span.0 + u_span.2 * iu;
                (0..=divs_v).map(move |iv| {
                    let iv = T::from_usize(iv).unwrap();
                    let v = v_span.0 + v_span.2 * iv;
                    Vector2::new(u, v)
                })
            })
            .collect();

        SurfaceTessellation {
            points,
            normals,
            faces,
            uvs,
        }
    }

    /// Evaluate the normal at the given u, v parameters
    pub fn normal_at(&self, u: T, v: T) -> OVector<T, DimNameDiff<D, U1>> {
        let deriv = self.rational_derivatives(u, v, 1);
        let v0 = &deriv[1][0];
        let v1 = &deriv[0][1];
        v0.cross(v1)
    }

    /// Evaluate the point and normal at the given u, v parameters
    #[allow(clippy::type_complexity)]
    pub fn point_normal_at(
        &self,
        u: T,
        v: T,
    ) -> (
        OPoint<T, DimNameDiff<D, U1>>,
        OVector<T, DimNameDiff<D, U1>>,
    ) {
        let deriv = self.rational_derivatives(u, v, 1);
        let point = deriv[0][0].clone();
        let normal = deriv[1][0].cross(&deriv[0][1]);
        (point.into(), normal)
    }

    /// Evaluate the rational derivatives at the given u, v parameters
    pub fn rational_derivatives(
        &self,
        u: T,
        v: T,
        derivs: usize,
    ) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> {
        let ders = self.derivatives(u, v, derivs);
        rational_derivatives(&ders, derivs)
    }

    /// Evaluate the derivatives at the given u, v parameters
    fn derivatives(&self, u: T, v: T, derivs: usize) -> Vec<Vec<OVector<T, D>>> {
        let n = self.u_knots.len() - self.u_degree - 2;
        let m = self.v_knots.len() - self.v_degree - 2;

        let du = if derivs < self.u_degree {
            derivs
        } else {
            self.u_degree
        };
        let dv = if derivs < self.v_degree {
            derivs
        } else {
            self.v_degree
        };
        let mut skl = vec![vec![OVector::<T, D>::zeros(); derivs + 1]; derivs + 1];
        let knot_span_index_u = self.u_knots.find_knot_span_index(n, self.u_degree, u);
        let knot_span_index_v = self.v_knots.find_knot_span_index(m, self.v_degree, v);
        let uders = self
            .u_knots
            .derivative_basis_functions(knot_span_index_u, u, self.u_degree, n);
        let vders = self
            .v_knots
            .derivative_basis_functions(knot_span_index_v, v, self.v_degree, m);
        let mut temp = vec![OPoint::<T, D>::origin(); self.v_degree + 1];

        for k in 0..=du {
            for s in 0..=self.v_degree {
                temp[s] = OPoint::<T, D>::origin();
                for r in 0..=self.u_degree {
                    let w = &self.control_points[knot_span_index_u - self.u_degree + r]
                        [knot_span_index_v - self.v_degree + s]
                        * uders[k][r];
                    let column = temp.get_mut(s).unwrap();
                    w.coords.iter().enumerate().for_each(|(i, v)| {
                        column[i] += *v;
                    });
                }
            }

            let nk = derivs - k;
            let dd = if nk < dv { nk } else { dv };

            for l in 0..=dd {
                for (s, item) in temp.iter().enumerate().take(self.v_degree + 1) {
                    let w = item * vders[l][s];
                    let column = skl[k].get_mut(l).unwrap();
                    w.coords.iter().enumerate().for_each(|(i, v)| {
                        column[i] += *v;
                    });
                }
            }
        }

        skl
    }

    /// Flip the surface in u or v direction or both
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let sphere = NurbsSurface3D::try_sphere(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.0).unwrap();
    /// let (ud, vd) = sphere.knots_domain();
    /// let p0 = sphere.point_at(ud.0, vd.0);
    ///
    /// let u_flipped = sphere.flip(FlipDirection::U);
    /// let (u_flipped_ud, u_flipped_vd) = u_flipped.knots_domain();
    /// let p1 = u_flipped.point_at(u_flipped_ud.1, u_flipped_vd.0);
    ///
    /// let v_flipped = sphere.flip(FlipDirection::V);
    /// let (v_flipped_ud, v_flipped_vd) = v_flipped.knots_domain();
    /// let p2 = v_flipped.point_at(v_flipped_ud.0, v_flipped_vd.1);
    /// assert_relative_eq!(p0, p2, epsilon = 1e-8);
    ///
    /// let uv_flipped = sphere.flip(FlipDirection::UV);
    /// let (uv_flipped_ud, uv_flipped_vd) = uv_flipped.knots_domain();
    /// let p3 = uv_flipped.point_at(uv_flipped_ud.1, uv_flipped_vd.1);
    /// assert_relative_eq!(p0, p3, epsilon = 1e-8);
    /// ```
    pub fn flip(&self, direction: FlipDirection) -> Self {
        let mut flipped = self.clone();

        // flip in u direction
        match direction {
            FlipDirection::U | FlipDirection::UV => {
                flipped.control_points = flipped.control_points.iter().rev().cloned().collect();
                flipped.u_knots = flipped.u_knots.inverse();
            }
            _ => {}
        }

        // flip in v direction
        match direction {
            FlipDirection::V | FlipDirection::UV => {
                flipped.control_points = flipped
                    .control_points
                    .iter()
                    .map(|row| row.iter().rev().cloned().collect())
                    .collect();
                flipped.v_knots = flipped.v_knots.inverse();
            }
            _ => {}
        }

        flipped
    }

    /// Create a plane shaped NURBS surface
    /// Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// let plane = NurbsSurface3D::<f64>::plane(Point3::origin(), Vector3::x(), Vector3::y());
    /// assert_eq!(plane.u_degree(), 1);
    /// assert_eq!(plane.v_degree(), 1);
    /// let (ud, vd) = plane.knots_domain();
    /// let p00 = plane.point_at(ud.0, vd.0);
    /// let p10 = plane.point_at(ud.1, vd.0);
    /// let p11 = plane.point_at(ud.1, vd.1);
    /// let p01 = plane.point_at(ud.0, vd.1);
    /// assert_eq!(p00, Point3::new(-1.0, -1.0, 0.0));
    /// assert_eq!(p10, Point3::new(1.0, -1.0, 0.0));
    /// assert_eq!(p11, Point3::new(1.0, 1.0, 0.0));
    /// assert_eq!(p01, Point3::new(-1.0, 1.0, 0.0));
    /// ```
    pub fn plane(
        center: OPoint<T, DimNameDiff<D, U1>>,
        x_axis: OVector<T, DimNameDiff<D, U1>>,
        y_axis: OVector<T, DimNameDiff<D, U1>>,
    ) -> Self
    where
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let control_points = vec![
            vec![
                OPoint::from_slice(
                    (center.clone() - x_axis.clone() - y_axis.clone())
                        .to_homogeneous()
                        .as_slice(),
                ),
                OPoint::from_slice(
                    (center.clone() - x_axis.clone() + y_axis.clone())
                        .to_homogeneous()
                        .as_slice(),
                ),
            ],
            vec![
                OPoint::from_slice(
                    (center.clone() + x_axis.clone() - y_axis.clone())
                        .to_homogeneous()
                        .as_slice(),
                ),
                OPoint::from_slice(
                    (center.clone() + x_axis + y_axis)
                        .to_homogeneous()
                        .as_slice(),
                ),
            ],
        ];

        Self {
            u_degree: 1,
            v_degree: 1,
            u_knots: KnotVector::new(vec![T::zero(), T::zero(), T::one(), T::one()]),
            v_knots: KnotVector::new(vec![T::zero(), T::zero(), T::one(), T::one()]),
            control_points,
        }
    }

    /// Extrude a NURBS curve to create a NURBS surface
    pub fn extrude(
        profile: &NurbsCurve<T, D>,
        translation: &OVector<T, DimNameDiff<D, U1>>,
    ) -> Self {
        let prof_points = profile.dehomogenized_control_points();
        let prof_weights = profile.weights();

        let half_translation = translation * T::from_f64(0.5).unwrap();

        let mut control_points = vec![vec![], vec![], vec![]];

        for i in 0..prof_points.len() {
            let p0 = &prof_points[i] + translation;
            let p1 = &prof_points[i] + &half_translation;
            let p2 = &prof_points[i];

            let mut o0 = OPoint::<T, D>::origin();
            let mut o1 = OPoint::<T, D>::origin();
            let mut o2 = OPoint::<T, D>::origin();
            let w = prof_weights[i];
            for j in 0..D::dim() - 1 {
                o0[j] = p0[j] * w;
                o1[j] = p1[j] * w;
                o2[j] = p2[j] * w;
            }
            o0[D::dim() - 1] = w;
            o1[D::dim() - 1] = w;
            o2[D::dim() - 1] = w;

            control_points[0].push(o0);
            control_points[1].push(o1);
            control_points[2].push(o2);
        }

        // dbg!(profile.control_points());
        // dbg!(&control_points);

        Self {
            u_degree: 2,
            v_degree: profile.degree(),
            u_knots: KnotVector::new(vec![
                T::zero(),
                T::zero(),
                T::zero(),
                T::one(),
                T::one(),
                T::one(),
            ]),
            v_knots: profile.knots().clone(),
            control_points,
        }
    }

    /// Try to loft a collection of curves to create a surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Translation3};
    ///
    /// // Create a collection of curves
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 2.0, 0.),
    ///     Point3::new(1.0, 2.5, 0.),
    /// ];
    /// let interpolated = NurbsCurve3D::interpolate(&points, 3).unwrap();
    /// // Offset the curve by translation of 2.0 in the z direction
    /// let offsetted = interpolated.transformed(&Translation3::new(0.0, 0.0, 2.0).into());
    ///
    /// // Loft the curves to create a NURBS surface
    /// let lofted = NurbsSurface::try_loft(&[interpolated, offsetted], None);
    /// assert!(lofted.is_ok());
    /// ```
    pub fn try_loft(curves: &[NurbsCurve<T, D>], degree_v: Option<usize>) -> anyhow::Result<Self>
    where
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<D, U1>>,
    {
        let unified_curves = try_unify_curve_knot_vectors(curves)?;

        let degree_u = unified_curves[0].degree();
        let degree_v = degree_v.unwrap_or(degree_u).min(unified_curves.len() - 1);

        let knots_u = unified_curves[0].knots().clone();

        // Interpolate each column of control points to get the nurbs curve aligned with the v direction
        let v_curves: anyhow::Result<Vec<_>> = (0..unified_curves[0].control_points().len())
            .map(|i| {
                let points = unified_curves
                    .iter()
                    .map(|c| c.control_points()[i].clone())
                    .collect_vec();
                let (control_points, knots) = try_interpolate_control_points(
                    &points
                        .iter()
                        .map(|p| DVector::from_vec(p.iter().copied().collect()))
                        .collect_vec(),
                    degree_v,
                    false,
                )?;
                NurbsCurve::try_new(
                    degree_v,
                    control_points
                        .iter()
                        .map(|p| OPoint::from_slice(p.as_slice()))
                        .collect(),
                    knots.to_vec(),
                )
            })
            .collect();
        let v_curves = v_curves?;
        let knots_v = v_curves.last().unwrap().knots().clone();
        let control_points = v_curves
            .iter()
            .map(|c| c.control_points().clone())
            .collect();

        Ok(Self {
            control_points,
            u_degree: degree_u,
            v_degree: degree_v,
            u_knots: knots_u,
            v_knots: knots_v,
        })
    }

    /// Try to create a iso curve from the surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let circle = NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    /// let extruded = NurbsSurface3D::extrude(&circle, &Vector3::z());
    ///
    /// // Create an iso curve at the start of the u direction
    /// let (start, _) = extruded.u_knots_domain();
    /// let u_iso = extruded.try_isocurve(start, UVDirection::U).unwrap();
    /// let (iso_start, iso_end) = u_iso.knots_domain();
    /// assert_relative_eq!(u_iso.point_at(iso_start), Point3::new(1.0, 0.0, 1.0), epsilon = 1e-8);
    /// assert_relative_eq!(u_iso.point_at(iso_end), Point3::new(1.0, 0.0, 1.0), epsilon = 1e-8);
    ///
    /// // Create an iso curve at the start of the v direction
    /// let (start, _) = extruded.v_knots_domain();
    /// let v_iso = extruded.try_isocurve(start, UVDirection::V).unwrap();
    /// let (iso_start, iso_end) = v_iso.knots_domain();
    /// assert_relative_eq!(v_iso.point_at(iso_start), Point3::new(1.0, 0.0, 1.0), epsilon = 1e-8);
    /// assert_relative_eq!(v_iso.point_at(iso_end), Point3::new(1.0, 0.0, 0.0), epsilon = 1e-8);
    /// ```
    pub fn try_isocurve(&self, t: T, direction: UVDirection) -> anyhow::Result<NurbsCurve<T, D>> {
        let (knots, degree) = match direction {
            UVDirection::U => (self.u_knots.clone(), self.u_degree),
            UVDirection::V => (self.v_knots.clone(), self.v_degree),
        };

        let mult = knots.multiplicity();
        let knots_to_insert = mult
            .iter()
            .enumerate()
            .find_map(|(i, m)| {
                if (t - *m.knot()).abs() < T::default_epsilon() {
                    Some(i)
                } else {
                    None
                }
            })
            .and_then(|knot_index| {
                let m = mult[knot_index].multiplicity();
                if degree + 1 >= m {
                    Some((degree + 1) - m)
                } else {
                    None
                }
            })
            .unwrap_or(degree + 1);

        let refined = if knots_to_insert > 0 {
            let mut refined = self.clone();
            refined.try_refine_knot(vec![t; knots_to_insert], direction)?;
            Cow::Owned(refined)
        } else {
            Cow::Borrowed(self)
        };

        let span = if (t - knots.first()).abs() < T::default_epsilon() {
            0
        } else if (t - knots.last()).abs() < T::default_epsilon() {
            match direction {
                UVDirection::U => refined.control_points.len() - 1,
                UVDirection::V => refined.control_points[0].len() - 1,
            }
        } else {
            knots.find_knot_span_index(knots.len() - degree - 2, degree, t)
        };

        match direction {
            UVDirection::U => NurbsCurve::try_new(
                refined.v_degree,
                refined.control_points[span].clone(),
                refined.v_knots.clone().to_vec(),
            ),
            UVDirection::V => NurbsCurve::try_new(
                refined.u_degree,
                refined
                    .control_points
                    .iter()
                    .map(|row| row[span].clone())
                    .collect(),
                refined.u_knots.clone().to_vec(),
            ),
        }
    }

    /// Try to create boundary curves of the surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let circle = NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    /// let extruded = NurbsSurface3D::extrude(&circle, &Vector3::z());
    /// let [c0, c1, c2, c3] = extruded.try_boundary_curves().unwrap();
    /// assert_relative_eq!(c0.point_at(c0.knots_domain().1), c1.point_at(c1.knots_domain().0), epsilon = 1e-8);
    /// assert_relative_eq!(c1.point_at(c1.knots_domain().1), c2.point_at(c2.knots_domain().0), epsilon = 1e-8);
    /// assert_relative_eq!(c2.point_at(c2.knots_domain().1), c3.point_at(c3.knots_domain().0), epsilon = 1e-8);
    /// assert_relative_eq!(c3.point_at(c3.knots_domain().1), c0.point_at(c0.knots_domain().0), epsilon = 1e-8);
    /// ```
    pub fn try_boundary_curves(&self) -> anyhow::Result<[NurbsCurve<T, D>; 4]> {
        let (u_start, u_end) = self.u_knots_domain();
        let (v_start, v_end) = self.v_knots_domain();
        Ok([
            // clockwise order
            self.try_isocurve(u_start, UVDirection::U)?, // v along (00 to 01)
            self.try_isocurve(v_end, UVDirection::V)?,   // u along (01 to 11)
            self.try_isocurve(u_end, UVDirection::U)?.inverse(), // v along (11 to 10)
            self.try_isocurve(v_start, UVDirection::V)?.inverse(), // u along (10 to 00)
        ])
    }

    /// Try to refine the surface by inserting knots
    pub fn try_refine_knot(
        &mut self,
        knots_to_insert: Vec<T>,
        direction: UVDirection,
    ) -> anyhow::Result<()> {
        match direction {
            UVDirection::U => {
                let refined = self
                    .transposed_control_points()
                    .iter()
                    .map(|row| {
                        let mut curve = NurbsCurve::try_new(
                            self.u_degree,
                            row.clone(),
                            self.u_knots.clone().to_vec(),
                        )?;
                        curve.try_refine_knot(knots_to_insert.clone())?;
                        Ok(curve)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;

                let u_knots = refined
                    .first()
                    .map(|c| c.knots().clone())
                    .ok_or(anyhow::anyhow!("No curves"))?;
                self.control_points = transpose_control_points(
                    &refined
                        .iter()
                        .map(|c| c.control_points().clone())
                        .collect_vec(),
                );
                self.u_knots = u_knots;
            }
            UVDirection::V => {
                let refined = self
                    .control_points
                    .iter()
                    .map(|row| {
                        let mut curve = NurbsCurve::try_new(
                            self.v_degree,
                            row.clone(),
                            self.v_knots.clone().to_vec(),
                        )?;
                        curve.try_refine_knot(knots_to_insert.clone())?;
                        Ok(curve)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;

                let v_knots = refined
                    .first()
                    .map(|c| c.knots().clone())
                    .ok_or(anyhow::anyhow!("No curves"))?;
                self.control_points = refined.iter().map(|c| c.control_points().clone()).collect();
                self.v_knots = v_knots;
            }
        };

        Ok(())
    }

    /// Find the closest point on the surface to a given point
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let sphere = NurbsSurface3D::try_sphere(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.0).unwrap();
    /// let ud = sphere.u_knots_domain();
    /// let vd = sphere.v_knots_domain();
    /// let div = 10;
    /// (0..div).for_each(|u| {
    ///     (0..div).for_each(|v| {
    ///         let u = u as f64 / div as f64 * (ud.1 - ud.0) + ud.0;
    ///         let v = v as f64 / div as f64 * (vd.1 - vd.0) + vd.0;
    ///         // println!("u: {}, v: {}", u, v);
    ///         let pt = sphere.point_at(u, v);
    ///         let pt2 = pt * 5.;
    ///         let closest = sphere.find_closest_point(&pt2, None).unwrap();
    ///         assert_relative_eq!(pt, closest, epsilon = 1e-4);
    ///     });
    /// });
    /// ```
    pub fn find_closest_point(
        &self,
        point: &OPoint<T, DimNameDiff<D, U1>>,
        hint: Option<(T, T)>,
    ) -> anyhow::Result<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        self.find_closest_parameter(point, hint)
            .map(|(u, v)| self.point_at(u, v))
    }

    /// Find the closest parameter on the surface to a given point with Newton's method
    pub fn find_closest_parameter(
        &self,
        point: &OPoint<T, DimNameDiff<D, U1>>,
        hint: Option<(T, T)>,
    ) -> anyhow::Result<(T, T)>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let uv = match hint {
            Some(uv) => Vector2::new(uv.0, uv.1),
            None => {
                let mut uv = Vector2::new(self.u_knots_domain().0, self.v_knots_domain().0);
                let mut min_dist = T::infinity();
                let tess = self.tessellate(Some(
                    AdaptiveTessellationOptions::<T, D>::default()
                        .with_min_divs_u((self.control_points().len() - 1) * 2)
                        .with_min_divs_v((self.control_points()[0].len() - 1) * 2),
                ));
                tess.points().iter().enumerate().for_each(|(i, pt)| {
                    let d = point - pt;
                    let d = d.norm_squared();
                    if d < min_dist {
                        min_dist = d;
                        uv = tess.uvs()[i];
                    }
                });
                uv
            }
        };
        // println!("Initial guess: {:?}", uv);

        let pts = self.dehomogenized_control_points();
        let u0 = pts.first().unwrap();
        let u1 = pts.last().unwrap();
        let eps = T::default_epsilon() * T::from_f64(10.0).unwrap();
        let u_closed = (0..pts[0].len()).all(|i| (&u0[i] - &u1[i]).norm() < eps);
        let v_closed = (0..pts.len()).all(|i| {
            let row = &pts[i];
            let v0 = row.first().unwrap();
            let v1 = row.last().unwrap();
            (v0 - v1).norm() < eps
        });
        // println!("u_closed: {}, v_closed: {}", u_closed, v_closed);

        let solver = SurfaceClosestParameterNewton::<T, D>::new(
            (self.u_knots_domain(), self.v_knots_domain()),
            (u_closed, v_closed),
        );
        let res = Executor::new(SurfaceClosestParameterProblem::new(point, self), solver)
            .configure(|state| state.param(uv).max_iters(100))
            .run()?;
        match res.state().get_best_param().cloned() {
            Some(t) => {
                if t.x.is_finite() && t.y.is_finite() {
                    Ok((t.x, t.y))
                } else {
                    Err(anyhow::anyhow!("No best parameter found"))
                }
            }
            _ => Err(anyhow::anyhow!("No best parameter found")),
        }
    }

    /// Cast the surface to a surface with another floating point type
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> NurbsSurface<F, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        NurbsSurface {
            control_points: self
                .control_points
                .iter()
                .map(|row| row.iter().map(|p| p.clone().cast()).collect())
                .collect(),
            u_degree: self.u_degree,
            v_degree: self.v_degree,
            u_knots: self.u_knots.cast(),
            v_knots: self.v_knots.cast(),
        }
    }
}

/// Compute the rational derivatives of derivatives
fn rational_derivatives<T, D>(
    ders: &[Vec<OVector<T, D>>],
    derivs: usize,
) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>>
where
    T: FloatingPoint,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let a_ders: Vec<_> = ders
        .iter()
        .map(|row| {
            row.iter()
                .map(|d| {
                    let mut a_ders = vec![];
                    for i in 0..D::dim() - 1 {
                        a_ders.push(d[i]);
                    }
                    OVector::<T, DimNameDiff<D, U1>>::from_vec(a_ders)
                })
                .collect_vec()
        })
        .collect();
    let w_ders: Vec<_> = ders
        .iter()
        .map(|row| row.iter().map(|d| d[D::dim() - 1]).collect_vec())
        .collect();

    let mut skl: Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> = vec![];
    let mut binom = Binomial::<T>::new();

    for k in 0..=derivs {
        let mut row = vec![];

        for l in 0..=(derivs - k) {
            let mut v = a_ders[k][l].clone();
            for j in 1..=l {
                let coef = binom.get(l, j) * w_ders[0][j];
                v -= &row[l - j] * coef;
            }

            for i in 1..=k {
                let coef = binom.get(k, i) * w_ders[i][0];
                v -= &skl[k - i][l] * coef;
                let mut v2 = OVector::<T, DimNameDiff<D, U1>>::zeros();
                for j in 1..=l {
                    v2 += &skl[k - i][l - j] * binom.get(l, j) * w_ders[i][j];
                }
                v -= v2 * binom.get(k, i);
            }

            let v = v / w_ders[0][0];
            row.push(v);
        }

        skl.push(row);
    }

    skl
}

/// A specialized trait for NURBS surfaces with 3D points,
/// particularly designed for sweeping operations that require Frenet frames.
impl<T: FloatingPoint> NurbsSurface3D<T> {
    /// Try to sweep a profile curve along a rail curve to create a surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Translation3};
    ///
    /// // Create a collection of curves
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let profile = NurbsCurve3D::interpolate(&points, 3).unwrap();
    ///
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(1.0, 0., 0.),
    ///     Point3::new(0.0, -1., 1.),
    ///     Point3::new(-1.0, 0., 2.),
    ///     Point3::new(0.0, 1., 3.),
    /// ];
    /// let rail= NurbsCurve3D::interpolate(&points, 3).unwrap();
    ///
    /// // Sweep the profile curve along the rail curve to create a NURBS surface
    /// let swept = NurbsSurface::try_sweep(&profile, &rail, Some(3));
    /// assert!(swept.is_ok());
    /// ```
    pub fn try_sweep(
        profile: &NurbsCurve3D<T>,
        rail: &NurbsCurve3D<T>,
        degree_v: Option<usize>,
    ) -> anyhow::Result<Self> {
        let (start, end) = rail.knots_domain();
        let samples = rail.control_points().len() * 2;
        let span = (end - start) / T::from_usize(samples - 1).unwrap();

        let parameters: Vec<_> = (0..samples)
            .map(|i| start + T::from_usize(i).unwrap() * span)
            .collect();

        let frames = rail.compute_frenet_frames(&parameters);
        let curves: Vec<_> = frames
            .iter()
            .map(|frame| {
                let transform = frame.matrix();
                profile.transformed(&transform.into())
            })
            .collect();

        Self::try_loft(&curves, degree_v)
    }

    /// Try to revolve a profile curve around an axis to create a surface
    /// /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    ///
    /// // Create a profile curve to revolve
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let profile = NurbsCurve3D::interpolate(&points, 3).unwrap();
    ///
    /// // Revolve the profile curve around the z-axis by PI radians to create a NURBS surface
    /// let revolved = NurbsSurface::try_revolve(&profile, &Point3::origin(), &Vector3::z_axis(), std::f64::consts::PI);
    /// assert!(revolved.is_ok());
    /// ```
    pub fn try_revolve(
        profile: &NurbsCurve3D<T>,
        center: &Point3<T>,
        axis: &Vector3<T>,
        theta: T,
    ) -> anyhow::Result<Self> {
        Self::try_revolve_by_start_end(profile, center, axis, T::zero(), theta)
    }

    /// Try to revolve a profile curve around an axis by specifying the start and end angles
    pub fn try_revolve_by_start_end(
        profile: &NurbsCurve3D<T>,
        center: &Point3<T>,
        axis: &Vector3<T>,
        start: T,
        end: T,
    ) -> anyhow::Result<Self> {
        let theta = end - start;
        let prof_points = profile.dehomogenized_control_points();
        let prof_weights = profile.weights();

        let two = T::from_f64(2.0).unwrap();
        let (narcs, mut u_knots) = if theta <= T::pi() / two {
            (1, vec![T::zero(); 6])
        } else if theta <= T::pi() {
            let mut knots = vec![T::zero(); 6 + 2];
            let half = T::from_f64(0.5).unwrap();
            knots[3] = half;
            knots[4] = half;
            (2, knots)
        } else if theta <= T::from_f64(3.0).unwrap() * T::pi() / two {
            let mut knots = vec![T::zero(); 6 + 2 * 2];
            let frac_three = T::from_f64(1.0 / 3.0).unwrap();
            let two_frac_three = T::from_f64(2.0 / 3.0).unwrap();
            knots[3] = frac_three;
            knots[4] = frac_three;
            knots[5] = two_frac_three;
            knots[6] = two_frac_three;
            (3, knots)
        } else {
            let mut knots = vec![T::zero(); 6 + 2 * 3];
            let frac_four = T::from_f64(1.0 / 4.0).unwrap();
            let half = T::from_f64(0.5).unwrap();
            let three_frac_four = T::from_f64(3.0 / 4.0).unwrap();
            knots[3] = frac_four;
            knots[4] = frac_four;
            knots[5] = half;
            knots[6] = half;
            knots[7] = three_frac_four;
            knots[8] = three_frac_four;
            (4, knots)
        };

        let dtheta = theta / T::from_usize(narcs).unwrap();
        let j = 3 + 2 * (narcs - 1);

        for i in 0..3 {
            u_knots[i] = T::zero();
            u_knots[j + i] = T::one();
        }

        let wm = (dtheta / two).cos();

        let angles = (0..=narcs)
            .map(|i| start + T::from_usize(i).unwrap() * dtheta)
            .collect_vec();
        let sines = angles.iter().map(|a| a.sin()).collect_vec();
        let cosines = angles.iter().map(|a| a.cos()).collect_vec();

        let mut control_points = vec![vec![Point4::origin(); prof_points.len()]; 2 * narcs + 1];

        for j in 0..prof_points.len() {
            let p = &prof_points[j];
            let s = (p - center).dot(axis);
            let o = center + axis * s;

            // vector from the axis
            let mut x = p - o;
            // radius at height
            let r = x.norm();
            // perpendicular vector to x & axis
            let mut y = axis.cross(&x);

            if r > T::default_epsilon() {
                x *= T::one() / r;
                y *= T::one() / r;
            }

            let mut p0 = prof_points[j];
            control_points[0][j].x = p0.x;
            control_points[0][j].y = p0.y;
            control_points[0][j].z = p0.z;
            control_points[0][j].w = prof_weights[j];

            let mut t0 = y;
            let mut index = 0;
            for i in 1..=narcs {
                let p2 = if r <= T::default_epsilon() {
                    o
                } else {
                    o + x * cosines[i] * r + y * sines[i] * r
                };

                let k = index + 2;
                control_points[k][j].x = p2.x;
                control_points[k][j].y = p2.y;
                control_points[k][j].z = p2.z;
                control_points[k][j].w = prof_weights[j];

                let t2 = y * cosines[i] - x * sines[i];

                let l = index + 1;
                if r <= T::default_epsilon() {
                    control_points[l][j].x = o.x;
                    control_points[l][j].y = o.y;
                    control_points[l][j].z = o.z;
                } else {
                    let nt0 = t0.normalize();
                    let nt2 = t2.normalize();
                    let r0 = Ray::new(p0, nt0);
                    let r1 = Ray::new(p2, nt2);
                    let intersection = r0.find_intersection(&r1).ok_or(anyhow::anyhow!(
                        "No intersection between rays: {:?}, {:?}",
                        r0,
                        r1
                    ))?;

                    let intersected = intersection.intersection0.0;
                    control_points[l][j].x = intersected.x;
                    control_points[l][j].y = intersected.y;
                    control_points[l][j].z = intersected.z;
                }

                control_points[l][j].w = wm * prof_weights[j];

                index += 2;

                if i < narcs {
                    p0 = p2;
                    t0 = t2;
                }
            }
        }

        // Scale the control points by their weights to make them rational
        control_points.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|p| {
                let w = p.w;
                p.x *= w;
                p.y *= w;
                p.z *= w;
            });
        });

        Ok(Self {
            control_points,
            u_degree: 2,
            v_degree: profile.degree(),
            u_knots: KnotVector::new(u_knots),
            v_knots: profile.knots().clone(),
        })
    }

    /// Try to create a sphere
    pub fn try_sphere(
        center: &Point3<T>,
        axis: &Vector3<T>,
        x_axis: &Vector3<T>,
        radius: T,
    ) -> anyhow::Result<Self> {
        let arc = NurbsCurve::try_arc(center, &-axis, x_axis, radius, T::zero(), T::pi())?;
        Self::try_revolve(&arc, center, axis, T::two_pi())
    }

    /// Try to create a torus (or partial torus)
    ///
    /// # Arguments
    /// * `center` - Center point of the torus
    /// * `x_axis` - X axis direction (will be normalized)
    /// * `y_axis` - Y axis direction (will be normalized)
    /// * `major_radius` - Distance from center to tube center
    /// * `minor_radius` - Radius of the tube
    /// * `u_domain` - Rotation angle range (major circle), in radians
    /// * `v_domain` - Profile arc angle range (minor circle), in radians
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    ///
    /// // Full torus
    /// let torus = NurbsSurface3D::try_torus(
    ///     &Point3::origin(),
    ///     &Vector3::x(),
    ///     &Vector3::y(),
    ///     2.0,
    ///     0.5,
    ///     (0.0, std::f64::consts::TAU),
    ///     (0.0, std::f64::consts::TAU)
    /// ).unwrap();
    /// let (u_start, u_end) = torus.u_knots_domain();
    /// let (v_start, v_end) = torus.v_knots_domain();
    /// assert_eq!(u_start, 0.0);
    /// assert_relative_eq!(u_end, std::f64::consts::TAU, epsilon = 1e-8);
    /// assert_eq!(v_start, 0.0);
    /// assert_relative_eq!(v_end, std::f64::consts::TAU, epsilon = 1e-8);
    ///
    /// // Partial torus (half rotation, half profile)
    /// let partial = NurbsSurface3D::try_torus(
    ///     &Point3::origin(),
    ///     &Vector3::x(),
    ///     &Vector3::y(),
    ///     2.0,
    ///     0.5,
    ///     (0.0, std::f64::consts::PI),
    ///     (0.0, std::f64::consts::PI)
    /// ).unwrap();
    /// ```
    pub fn try_torus(
        center: &Point3<T>,
        x_axis: &Vector3<T>,
        y_axis: &Vector3<T>,
        major_radius: T,
        minor_radius: T,
        u_domain: (T, T),
        v_domain: (T, T),
    ) -> anyhow::Result<Self> {
        // Normalize the axes
        let x_axis = x_axis.normalize();
        let y_axis = y_axis.normalize();
        let z_axis = x_axis.cross(&y_axis).normalize();

        let delta_u = u_domain.1 - u_domain.0;
        let delta_v = v_domain.1 - v_domain.0;

        anyhow::ensure!(
            delta_u > T::zero() && delta_u <= T::two_pi(),
            "u_domain must be in range (0, 2π]"
        );
        anyhow::ensure!(
            delta_v > T::zero() && delta_v <= T::two_pi(),
            "v_domain must be in range (0, 2π]"
        );

        // Number of spans: match try_arc logic
        // <= 90°: 1 span, <= 180°: 2 spans, > 180°: 4 spans
        let half_pi = T::pi() / T::from_f64(2.0).unwrap();
        let nb_u_spans = if delta_u <= half_pi + T::from_f64(1e-8).unwrap() {
            1
        } else if delta_u <= T::pi() + T::from_f64(1e-8).unwrap() {
            2
        } else {
            4
        };

        let nb_v_spans = if delta_v <= half_pi + T::from_f64(1e-8).unwrap() {
            1
        } else if delta_v <= T::pi() + T::from_f64(1e-8).unwrap() {
            2
        } else {
            4
        };

        let two = T::from_f64(2.0).unwrap();
        let alfa_u = delta_u / (T::from_usize(nb_u_spans).unwrap() * two);
        let alfa_v = delta_v / (T::from_usize(nb_v_spans).unwrap() * two);

        let nb_u_poles = 2 * nb_u_spans + 1;
        let nb_v_poles = 2 * nb_v_spans + 1;

        // Compute V-direction control points (minor circle profile in XZ plane)
        // x[j] = distance from Z axis, z[j] = height
        let mut x_coords = vec![T::zero(); nb_v_poles];
        let mut z_coords = vec![T::zero(); nb_v_poles];

        x_coords[0] = major_radius + minor_radius * v_domain.0.cos();
        z_coords[0] = minor_radius * v_domain.0.sin();

        let mut v_start = v_domain.0;
        for i in 1..=nb_v_spans {
            let cos_alfa_v = alfa_v.cos();
            // Odd index: tangent intersection point
            x_coords[2 * i - 1] =
                major_radius + minor_radius * (v_start + alfa_v).cos() / cos_alfa_v;
            z_coords[2 * i - 1] = minor_radius * (v_start + alfa_v).sin() / cos_alfa_v;
            // Even index: point on circle
            x_coords[2 * i] = major_radius + minor_radius * (v_start + two * alfa_v).cos();
            z_coords[2 * i] = minor_radius * (v_start + two * alfa_v).sin();
            v_start += two * alfa_v;
        }

        // Build control points by rotating V-profile around Z axis
        let mut control_points = vec![vec![Point4::origin(); nb_v_poles]; nb_u_poles];

        // First column (u = u_domain.0)
        let u_start_cos = u_domain.0.cos();
        let u_start_sin = u_domain.0.sin();
        for j in 0..nb_v_poles {
            let p = center
                + x_axis * (x_coords[j] * u_start_cos)
                + y_axis * (x_coords[j] * u_start_sin)
                + z_axis * z_coords[j];
            control_points[0][j] = Point4::new(p.x, p.y, p.z, T::one());
        }

        // Fill remaining columns
        let cos_alfa_u = alfa_u.cos();
        let mut u_angle = u_domain.0;
        for i in 1..=nb_u_spans {
            for j in 0..nb_v_poles {
                // Middle point of span (odd u index): tangent intersection point
                let u1 = u_angle + alfa_u;
                let cos_u1 = u1.cos();
                let sin_u1 = u1.sin();
                // Dehomogenized position divided by cos(alfa_u)
                let x1 = x_coords[j] * cos_u1 / cos_alfa_u;
                let y1 = x_coords[j] * sin_u1 / cos_alfa_u;
                let z1 = z_coords[j];
                let p1 = center + x_axis * x1 + y_axis * y1 + z_axis * z1;
                // Store with weight cos(alfa_u)
                control_points[2 * i - 1][j] = Point4::new(
                    p1.x * cos_alfa_u,
                    p1.y * cos_alfa_u,
                    p1.z * cos_alfa_u,
                    cos_alfa_u,
                );

                // End point of span (even u index): point on circle
                let u2 = u_angle + two * alfa_u;
                let cos_u2 = u2.cos();
                let sin_u2 = u2.sin();
                let x2 = x_coords[j] * cos_u2;
                let y2 = x_coords[j] * sin_u2;
                let z2 = z_coords[j];
                let p2 = center + x_axis * x2 + y_axis * y2 + z_axis * z2;
                control_points[2 * i][j] = Point4::new(p2.x, p2.y, p2.z, T::one());
            }
            u_angle += two * alfa_u;
        }

        // Set weights for V-direction (already incorporated in control points for U)
        let cos_alfa_v = alfa_v.cos();
        for i in 0..nb_u_poles {
            for j in 0..nb_v_poles {
                if j % 2 == 1 {
                    // Odd v index: multiply weight by cos(alfa_v)
                    let w = control_points[i][j].w * cos_alfa_v;
                    control_points[i][j].x = control_points[i][j].x / control_points[i][j].w * w;
                    control_points[i][j].y = control_points[i][j].y / control_points[i][j].w * w;
                    control_points[i][j].z = control_points[i][j].z / control_points[i][j].w * w;
                    control_points[i][j].w = w;
                }
            }
        }

        // Build knot vectors matching try_arc's approach
        let mut u_knots = Vec::new();
        let mut v_knots = Vec::new();

        let angle_per_u_span = delta_u / T::from_usize(nb_u_spans).unwrap();
        let angle_per_v_span = delta_v / T::from_usize(nb_v_spans).unwrap();

        // U knots: degree+1 copies of first
        let degree = 2;
        for _ in 0..=degree {
            u_knots.push(u_domain.0);
        }

        // Internal knots (each with multiplicity 2)
        for i in 1..nb_u_spans {
            let knot = u_domain.0 + angle_per_u_span * T::from_usize(i).unwrap();
            u_knots.push(knot);
            u_knots.push(knot);
        }

        // degree+1 copies of last
        for _ in 0..=degree {
            u_knots.push(u_domain.1);
        }

        // V knots: same structure
        for _ in 0..=degree {
            v_knots.push(v_domain.0);
        }

        for i in 1..nb_v_spans {
            let knot = v_domain.0 + angle_per_v_span * T::from_usize(i).unwrap();
            v_knots.push(knot);
            v_knots.push(knot);
        }

        for _ in 0..=degree {
            v_knots.push(v_domain.1);
        }

        let torus = Self {
            control_points,
            u_degree: 2,
            v_degree: 2,
            u_knots: KnotVector::new(u_knots),
            v_knots: KnotVector::new(v_knots),
        };

        Ok(torus)
    }
}

/// Unify the knot vectors of a collection of NURBS curves
///
fn try_unify_curve_knot_vectors<T, D>(
    curves: &[NurbsCurve<T, D>],
) -> anyhow::Result<Vec<NurbsCurve<T, D>>>
where
    T: FloatingPoint,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    let max_degree = curves.iter().fold(0, |d, c| d.max(c.degree()));

    // elevate all curves to the same degree
    let mut curves = curves
        .iter()
        .map(|c| {
            if c.degree() < max_degree {
                c.try_elevate_degree(max_degree)
            } else {
                Ok(c.clone())
            }
        })
        .collect::<anyhow::Result<Vec<NurbsCurve<T, D>>>>()?;

    let knot_intervals = curves
        .iter()
        .map(|c| {
            let knots = c.knots();
            let min = knots[0];
            let max = knots[knots.len() - 1];
            (min, max)
        })
        .collect_vec();

    //shift all knot vectors to start at 0.0
    curves.iter_mut().enumerate().for_each(|(i, c)| {
        let min = knot_intervals[i].0;
        c.knots_mut().iter_mut().for_each(|x| *x -= min);
    });

    //find the max knot span
    let knot_spans = knot_intervals
        .iter()
        .map(|(min, max)| *max - *min)
        .collect_vec();
    let max_knot_span = knot_spans.iter().fold(T::zero(), |x, a| a.max(x));

    //scale all of the knot vectors to match
    curves.iter_mut().enumerate().for_each(|(i, c)| {
        let scale = max_knot_span / knot_spans[i];
        c.knots_mut().iter_mut().for_each(|x| *x *= scale);
    });

    //merge all of the knot vectors
    let merged_knots = curves
        .iter()
        .fold(vec![], |a, c| sorted_set_union(c.knots().as_slice(), &a));

    //knot refinement on each curve
    for curve in curves.iter_mut() {
        let rem = sorted_set_sub(&merged_knots, curve.knots().as_slice());
        if !rem.is_empty() {
            curve.try_refine_knot(rem)?;
        }
    }

    Ok(curves)
}

fn sorted_set_union<T: RealField + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut merged = Vec::new();
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() || bi < b.len() {
        if ai >= a.len() {
            merged.push(b[bi]);
            bi += 1;
            continue;
        } else if bi >= b.len() {
            merged.push(a[ai]);
            ai += 1;
            continue;
        }

        let diff = a[ai] - b[bi];

        if diff.abs() < T::default_epsilon() {
            merged.push(a[ai]);
            ai += 1;
            bi += 1;
            continue;
        }

        if diff > T::zero() {
            // add the smallar value
            merged.push(b[bi]);
            bi += 1;
            continue;
        }

        // thus diff < 0.0
        merged.push(a[ai]);
        ai += 1;
    }

    merged
}

fn sorted_set_sub<T: RealField + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut ai = 0;
    let mut bi = 0;

    while ai < a.len() {
        if bi >= b.len() {
            result.push(a[ai]);
            ai += 1;
            continue;
        }

        if (a[ai] - b[bi]).abs() < T::default_epsilon() {
            ai += 1;
            bi += 1;
            continue;
        }

        result.push(a[ai]);
        ai += 1;
    }

    result
}

/// Enable to transform a NURBS surface by a given DxD matrix
impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for NurbsSurface<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.control_points.iter_mut().for_each(|rows| {
            rows.iter_mut().for_each(|p| {
                // dehomogenize
                let ow = p[D - 1];
                let mut pt = *p;
                for i in 0..D - 1 {
                    pt[i] /= ow;
                }

                pt[D - 1] = T::one();
                let transformed = transform * pt;
                let w = transformed[D - 1];
                for i in 0..D - 1 {
                    p[i] = transformed[i] / w * ow;
                }
            });
        });
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Reverse the direction of the surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let mut surface = NurbsSurface::try_sphere(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.0).unwrap();
    /// let (u_start, u_end) = surface.u_knots_domain();
    /// let (v_start, v_end) = surface.v_knots_domain();
    /// assert_relative_eq!(surface.point_at(u_start, v_start), Point3::new(-1.0, 0.0, 0.0), epsilon = 1e-8);
    /// assert_relative_eq!(surface.point_at(u_end, v_end), Point3::new(1.0, 0.0, 0.0), epsilon = 1e-8);
    /// surface.invert();
    /// assert_relative_eq!(surface.point_at(u_start, v_start), Point3::new(1.0, 0.0, 0.0), epsilon = 1e-8);
    /// assert_relative_eq!(surface.point_at(u_end, v_end), Point3::new(-1.0, 0.0, 0.0), epsilon = 1e-8);
    /// ```
    fn invert(&mut self) {
        self.control_points.iter_mut().for_each(|row| {
            row.reverse();
        });
        self.control_points.reverse();
        self.u_knots.invert();
        self.v_knots.invert();
    }
}

#[cfg(feature = "serde")]
impl<T, D: DimName> serde::Serialize for NurbsSurface<T, D>
where
    T: FloatingPoint + serde::Serialize,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NurbsSurface", 5)?;
        state.serialize_field("control_points", &self.control_points)?;
        state.serialize_field("u_degree", &self.u_degree)?;
        state.serialize_field("v_degree", &self.v_degree)?;
        state.serialize_field("u_knots", &self.u_knots)?;
        state.serialize_field("v_knots", &self.v_knots)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, D: DimName> serde::Deserialize<'de> for NurbsSurface<T, D>
where
    T: FloatingPoint + serde::Deserialize<'de>,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>: serde::Deserialize<'de>,
{
    fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
    where
        S: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};

        #[derive(Debug)]
        enum Field {
            ControlPoints,
            UDegree,
            VDegree,
            UKnots,
            VKnots,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
            where
                S: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("`control_points` or `u_degree` or `v_degree` or `u_knots` or `v_knots`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "control_points" => Ok(Field::ControlPoints),
                            "u_degree" => Ok(Field::UDegree),
                            "v_degree" => Ok(Field::VDegree),
                            "u_knots" => Ok(Field::UKnots),
                            "v_knots" => Ok(Field::VKnots),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct NurbsSurfaceVisitor<T, D>(std::marker::PhantomData<(T, D)>);

        impl<T, D> NurbsSurfaceVisitor<T, D> {
            pub fn new() -> Self {
                NurbsSurfaceVisitor(std::marker::PhantomData)
            }
        }

        impl<'de, T, D: DimName> Visitor<'de> for NurbsSurfaceVisitor<T, D>
        where
            T: FloatingPoint + serde::Deserialize<'de>,
            DefaultAllocator: Allocator<D>,
            <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>:
                serde::Deserialize<'de>,
        {
            type Value = NurbsSurface<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct NurbsSurface")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut control_points = None;
                let mut u_degree = None;
                let mut v_degree = None;
                let mut u_knots = None;
                let mut v_knots = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::ControlPoints => {
                            if control_points.is_some() {
                                return Err(de::Error::duplicate_field("control_points"));
                            }
                            control_points = Some(map.next_value()?);
                        }
                        Field::UDegree => {
                            if u_degree.is_some() {
                                return Err(de::Error::duplicate_field("u_degree"));
                            }
                            u_degree = Some(map.next_value()?);
                        }
                        Field::VDegree => {
                            if v_degree.is_some() {
                                return Err(de::Error::duplicate_field("v_degree"));
                            }
                            v_degree = Some(map.next_value()?);
                        }
                        Field::UKnots => {
                            if u_knots.is_some() {
                                return Err(de::Error::duplicate_field("u_knots"));
                            }
                            u_knots = Some(map.next_value()?);
                        }
                        Field::VKnots => {
                            if v_knots.is_some() {
                                return Err(de::Error::duplicate_field("v_knots"));
                            }
                            v_knots = Some(map.next_value()?);
                        }
                    }
                }
                let control_points =
                    control_points.ok_or_else(|| de::Error::missing_field("control_points"))?;
                let u_degree = u_degree.ok_or_else(|| de::Error::missing_field("u_degree"))?;
                let v_degree = v_degree.ok_or_else(|| de::Error::missing_field("v_degree"))?;
                let u_knots = u_knots.ok_or_else(|| de::Error::missing_field("u_knots"))?;
                let v_knots = v_knots.ok_or_else(|| de::Error::missing_field("v_knots"))?;

                Ok(Self::Value {
                    control_points,
                    u_degree,
                    v_degree,
                    u_knots,
                    v_knots,
                })
            }
        }

        const FIELDS: &[&str] = &[
            "control_points",
            "u_degree",
            "v_degree",
            "u_knots",
            "v_knots",
        ];
        deserializer.deserialize_struct("NurbsSurface", FIELDS, NurbsSurfaceVisitor::<T, D>::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_torus_basic() {
        let torus = NurbsSurface3D::<f64>::try_torus(
            &Point3::origin(),
            &Vector3::x(),
            &Vector3::y(),
            3.0,
            1.0,
            (0.0, std::f64::consts::TAU),
            (0.0, std::f64::consts::TAU),
        )
        .unwrap();

        // Check degrees
        assert_eq!(torus.u_degree(), 2);
        assert_eq!(torus.v_degree(), 2);

        // Check domains
        let (u_start, u_end) = torus.u_knots_domain();
        let (v_start, v_end) = torus.v_knots_domain();
        assert_eq!(u_start, 0.0);
        assert_relative_eq!(u_end, std::f64::consts::TAU, epsilon = 1e-8);
        assert_eq!(v_start, 0.0);
        assert_relative_eq!(v_end, std::f64::consts::TAU, epsilon = 1e-8);
    }
}
