use std::f64::consts::{FRAC_PI_2, TAU};
use std::vec;

use argmin::core::{ArgminFloat, Executor, State};
use gauss_quad::GaussLegendre;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub,
    OMatrix, OPoint, OVector, RealField, Rotation3, UnitVector3, Vector3, U1,
};
use simba::scalar::SupersetOf;

use crate::misc::binomial::Binomial;
use crate::misc::frenet_frame::FrenetFrame;
use crate::misc::transformable::Transformable;
use crate::misc::trigonometry::segment_closest_point;
use crate::misc::Ray;
use crate::prelude::{CurveLengthParameter, Invertible, KnotVector};
use crate::{misc::FloatingPoint, CurveClosestParameterNewton, CurveClosestParameterProblem};

use super::KnotStyle;

/// NURBS curve representation
/// By generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug, PartialEq)]
pub struct NurbsCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    /// control points with homogeneous coordinates
    /// the last element of the vector is the `weight`
    control_points: Vec<OPoint<T, D>>,
    degree: usize,
    /// knot vector for the NURBS curve
    /// the length of the knot vector is equal to the `# of control points + degree + 1`
    knots: KnotVector<T>,
}

/// 2D NURBS curve alias
pub type NurbsCurve2D<T> = NurbsCurve<T, Const<3>>;

/// 3D NURBS curve alias
pub type NurbsCurve3D<T> = NurbsCurve<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a new NURBS curve
    /// # Failures
    /// - if the number of control points is less than the degree
    /// - the number of knots is not equal to the number of control points + the degree + 1
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    ///
    /// let w = 1.; // weight for each control points
    /// let control_points: Vec<Point3<f64>> = vec![
    ///     Point3::new(50., 50., w),
    ///     Point3::new(30., 370., w),
    ///     Point3::new(180., 350., w),
    ///     Point3::new(150., 100., w),
    ///     Point3::new(250., 50., w),
    ///     Point3::new(350., 100., w),
    ///     Point3::new(470., 400., w),
    /// ];
    /// let degree = 3;
    /// let m = control_points.len() + degree + 1;
    /// // create an uniform knot vector
    /// let knots = (0..m).map(|i| i as f64).collect();
    /// let nurbs = NurbsCurve::try_new(3, control_points, knots);
    /// assert!(nurbs.is_ok());
    /// ```
    pub fn try_new(
        degree: usize,
        control_points: Vec<OPoint<T, D>>,
        knots: Vec<T>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            control_points.len() > degree,
            "Too few control points for curve"
        );
        anyhow::ensure!(
            knots.len() == control_points.len() + degree + 1,
            "Invalid number of knots, got {}, expected {}",
            knots.len(),
            control_points.len() + degree + 1
        );

        let mut knots = knots.clone();
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(Self {
            degree,
            control_points,
            knots: KnotVector::new(knots),
        })
    }

    /// Create a new NURBS curve without checking the validity of the control points and knots
    pub fn new_unchecked(
        degree: usize,
        control_points: Vec<OPoint<T, D>>,
        knots: KnotVector<T>,
    ) -> Self {
        Self {
            degree,
            control_points,
            knots,
        }
    }

    /// Create a new NURBS curve with a 1 degree as a polyline
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// use itertools::Itertools;
    /// let points = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 0.0),
    ///     Point2::new(-1.0, 0.0),
    ///     Point2::new(-1.0, 1.0),
    ///     Point2::new(1.0, 1.0),
    /// ];
    /// let polyline_curve = NurbsCurve2D::polyline(&points, true);
    /// let (start, end) = polyline_curve.knots_domain();
    /// let start = polyline_curve.point_at(start);
    /// let end = polyline_curve.point_at(end);
    /// assert_eq!(points[0], start);
    /// assert_eq!(points[points.len() - 1], end);
    ///
    /// let length = polyline_curve.try_length().unwrap();
    /// let goal = points.iter().tuple_windows().map(|(a, b)| (a - b).norm()).sum();
    /// assert_eq!(length, goal);
    /// ```
    pub fn polyline(points: &[OPoint<T, DimNameDiff<D, U1>>], normalize_knots: bool) -> Self
    where
        D: DimNameSub<U1>,
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let mut knots = vec![T::zero(); 2];

        let mut acc = T::zero();
        for i in 0..(points.len() - 1) {
            acc += (&points[i] - &points[i + 1]).norm();
            knots.push(acc);
        }
        knots.push(acc);

        let knots = if normalize_knots {
            knots.into_iter().map(|k| k / acc).collect()
        } else {
            knots
        };

        Self {
            degree: 1,
            knots: KnotVector::new(knots),
            control_points: points
                .iter()
                .map(|p| {
                    let coord = p.to_homogeneous();
                    OPoint::from_slice(coord.as_slice())
                })
                .collect(),
        }
    }

    /// Create a bezier curve from a list of control points
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// let points = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(-1.0, 1.0),
    /// ];
    /// let bezier_curve = NurbsCurve2D::bezier(&points);
    /// println!("bezier_curve: {:?}", bezier_curve);
    /// assert_eq!(bezier_curve.degree(), 3);
    /// assert_eq!(bezier_curve.knots().len(), 8);
    /// let domain = bezier_curve.knots_domain();
    /// assert_eq!(domain, (0.0, 1.0));
    /// let start = bezier_curve.point_at(domain.0);
    /// let end = bezier_curve.point_at(domain.1);
    /// assert_eq!(start, points[0]);
    /// assert_eq!(end, points[points.len() - 1]);
    /// ```
    pub fn bezier(points: &[OPoint<T, DimNameDiff<D, U1>>]) -> Self
    where
        D: DimNameSub<U1>,
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let degree = points.len() - 1;
        let knots = vec![T::zero(); degree + 1]
            .into_iter()
            .chain(vec![T::one(); degree + 1])
            .collect_vec();
        Self {
            degree,
            knots: KnotVector::new(knots),
            control_points: points
                .iter()
                .map(|p| {
                    let coord = p.to_homogeneous();
                    OPoint::from_slice(coord.as_slice())
                })
                .collect(),
        }
    }

    /// Create a dehomogenized version of the curve
    pub fn dehomogenize(&self) -> NurbsCurve<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        NurbsCurve {
            degree: self.degree,
            control_points: self.dehomogenized_control_points(),
            knots: self.knots.clone(),
        }
    }

    /// Return the dehomogenized control points
    pub fn dehomogenized_control_points(&self) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        self.control_points
            .iter()
            .map(|p| dehomogenize(p).unwrap())
            .collect()
    }

    pub fn weights(&self) -> Vec<T> {
        self.control_points
            .iter()
            .map(|p| p[D::dim() - 1])
            .collect()
    }

    /// Evaluate the curve at a given parameter to get a dehomonogenized point
    pub fn point_at(&self, t: T) -> OPoint<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let p = self.point(t);
        dehomogenize(&p).unwrap()
    }

    /// Sample the curve at a given number of points between the start and end parameters
    pub fn sample_regular_range(
        &self,
        start: T,
        end: T,
        samples: usize,
    ) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let mut points = vec![];
        let us = T::from_usize(samples).unwrap();
        let step = (end - start) / (us - T::one());
        for i in 0..samples {
            let t = start + T::from_usize(i).unwrap() * step;
            points.push(self.point_at(t));
        }
        points
    }

    #[allow(clippy::type_complexity)]
    /// Sample the curve at a given number of points between the start and end
    /// Return the vector of tuples of parameter and point
    pub fn sample_regular_range_with_parameter(
        &self,
        start: T,
        end: T,
        samples: usize,
    ) -> Vec<(T, OPoint<T, DimNameDiff<D, U1>>)>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let mut points = vec![];
        let us = T::from_usize(samples).unwrap();
        let step = (end - start) / (us - T::one());
        for i in 0..samples {
            let t = start + T::from_usize(i).unwrap() * step;
            points.push((t, self.point_at(t)));
        }
        points
    }

    /// Evaluate the curve at a given parameter to get a point
    pub(crate) fn point(&self, t: T) -> OPoint<T, D> {
        let n = self.knots.len() - self.degree - 2;
        let knot_span_index = self.knots.find_knot_span_index(n, self.degree, t);
        let basis = self.knots.basis_functions(knot_span_index, t, self.degree);
        let mut position = OPoint::<T, D>::origin();
        for i in 0..=self.degree {
            position.coords +=
                &self.control_points[knot_span_index - self.degree + i].coords * basis[i];
        }
        position
    }

    /// Evaluate the curve at a given parameter to get a tangent vector
    pub fn tangent_at(&self, u: T) -> OVector<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let deriv = self.rational_derivatives(u, 1);
        deriv[1].clone()
    }

    /// Compute second order derivative at a given parameter.
    pub fn second_derivative_at(&self, u: T) -> OVector<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let deriv = self.rational_derivatives(u, 2);
        deriv[2].clone()
    }

    /// Evaluate the curve at a given parameter to get a point & tangent vector at the same time
    #[allow(clippy::type_complexity)]
    pub fn point_tangent_at(
        &self,
        u: T,
    ) -> (
        OPoint<T, DimNameDiff<D, U1>>,
        OVector<T, DimNameDiff<D, U1>>,
    )
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let deriv = self.rational_derivatives(u, 1);
        (deriv[0].clone().into(), deriv[1].clone())
    }

    /// Evaluate the rational derivatives at a given parameter
    pub(crate) fn rational_derivatives(
        &self,
        u: T,
        derivs: usize,
    ) -> Vec<OVector<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let ders = self.derivatives(u, derivs);
        let a_ders: Vec<_> = ders
            .iter()
            .map(|d| {
                let mut a_ders = vec![];
                for i in 0..D::dim() - 1 {
                    a_ders.push(d[i]);
                }
                OVector::<T, DimNameDiff<D, U1>>::from_vec(a_ders)
            })
            .collect();
        let w_ders: Vec<_> = ders.iter().map(|d| d[D::dim() - 1]).collect();

        let mut ck = vec![];
        let mut binom = Binomial::<T>::new();
        for k in 0..=derivs {
            let mut v = a_ders[k].clone();

            for i in 1..=k {
                let coef = binom.get(k, i) * w_ders[i];
                v -= &ck[k - i] * coef;
            }

            let dehom = v / w_ders[0];
            ck.push(dehom);
        }
        ck
    }

    /// Evaluate the derivatives at a given parameter
    fn derivatives(&self, u: T, derivs: usize) -> Vec<OVector<T, D>> {
        let n = self.knots.len() - self.degree - 2;

        let du = if derivs < self.degree {
            derivs
        } else {
            self.degree
        };
        let mut derivatives = vec![OVector::<T, D>::zeros(); derivs + 1];

        let knot_span_index = self.knots.find_knot_span_index(n, self.degree, u);
        let nders = self
            .knots
            .derivative_basis_functions(knot_span_index, u, self.degree, du);
        for k in 0..=du {
            for j in 0..=self.degree {
                let w = &self.control_points[knot_span_index - self.degree + j] * nders[k][j];
                let column = derivatives.get_mut(k).unwrap();
                w.coords.iter().enumerate().for_each(|(i, v)| {
                    column[i] += *v;
                });
            }
        }

        derivatives
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn knots(&self) -> &KnotVector<T> {
        &self.knots
    }

    pub fn knots_mut(&mut self) -> &mut KnotVector<T> {
        &mut self.knots
    }

    pub fn control_points(&self) -> &Vec<OPoint<T, D>> {
        &self.control_points
    }

    pub fn control_points_iter(&self) -> impl Iterator<Item = &OPoint<T, D>> {
        self.control_points.iter()
    }

    pub fn control_points_iter_mut(&mut self) -> impl Iterator<Item = &mut OPoint<T, D>> {
        self.control_points.iter_mut()
    }

    pub fn knots_domain(&self) -> (T, T) {
        self.knots.domain(self.degree)
    }

    pub fn knots_domain_interval(&self) -> T {
        let (d0, d1) = self.knots_domain();
        d1 - d0
    }

    pub fn knots_constrain(&self, u: T) -> T {
        self.knots.clamp(self.degree, u)
    }

    /// Normalize the knot vector to the range of [0, 1]
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::TAU;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let (min, max) = unit_circle.knots_domain();
    /// assert_eq!(min, 0.);
    /// assert_eq!(max, TAU);
    /// let mut normalized = unit_circle.clone();
    /// normalized.normalize_knots();
    /// let (min, max) = normalized.knots_domain();
    /// assert_eq!(min, 0.);
    /// assert_eq!(max, 1.);
    /// ```
    pub fn normalize_knots(&mut self) {
        let knots = self.knots();
        let (min, max) = self.knots_domain();
        let size = max - min;
        let normalized = knots.iter().map(|k| (*k - min) / size).collect_vec();
        self.knots = KnotVector::new(normalized);
    }

    /// Compute the length of the curve by gauss-legendre quadrature
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let approx = unit_circle.try_length().unwrap();
    /// let goal = 2.0 * std::f64::consts::PI; // circumference of the unit circle
    /// assert_relative_eq!(approx, goal);
    /// ```
    pub fn try_length(&self) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let mult = self.knots.multiplicity();
        let start = mult.first().unwrap().multiplicity();
        let end = mult.last().unwrap().multiplicity();

        let segments = self.try_decompose_bezier_segments()?;

        // If the start/end parts of the knot vector are not duplicated,
        // the Bezier segments will not be generated correctly,
        // so reduce the number of segments by the amount that falls below the required duplication degree.
        let required_multiplicity = self.degree + 1;
        let i = required_multiplicity.saturating_sub(start);
        let j = if end < required_multiplicity {
            segments.len() - (required_multiplicity - end)
        } else {
            segments.len()
        };
        let segments = &segments[i..j];

        let (_, u) = self.knots_domain();
        let gauss = GaussLegendre::new(16 + self.degree)?;

        if segments.len() <= 1 {
            // If the curve is a single Bezier segment, compute the length directly
            let l = compute_bezier_segment_length(self, u, &gauss);
            return Ok(l);
        }

        let length = segments
            .iter()
            .map(|s| compute_bezier_segment_length(s, u, &gauss))
            .reduce(T::add);
        length.ok_or(anyhow::anyhow!("Failed to compute the length of the curve"))
    }

    /// Compute the parameter at a given length
    /// `tolerance` defines the precision of the result (default: 1e-4)
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let u = unit_circle.try_parameter_at_length(std::f64::consts::PI, Some(1e-4)).unwrap();
    /// assert_relative_eq!(u, std::f64::consts::PI, epsilon = 1e-4);
    ///
    /// let polyline = NurbsCurve2D::polyline(&[Point2::new(0., 0.), Point2::new(1., 0.), Point2::new(1., 1.)], false);
    /// let u = polyline.try_parameter_at_length(1.5, Some(1e-4)).unwrap();
    /// assert_relative_eq!(u, 1.5, epsilon = 1e-4);
    /// ```
    pub fn try_parameter_at_length(&self, length: T, tolerance: Option<T>) -> anyhow::Result<T>
    where
        D: DimName + DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        if length < T::default_epsilon() {
            return Ok(self.knots_domain().0);
        }

        let segments = self.try_decompose_bezier_segments()?;
        let gauss = GaussLegendre::new(16 + self.degree)?;
        let tolerance = tolerance.unwrap_or(T::from_f64(1e-4).unwrap());

        let mut acc_length = T::zero();

        for segment in segments {
            let (_, end) = segment.knots_domain();
            let segment_length = compute_bezier_segment_length(&segment, end, &gauss);

            let l = length - acc_length;
            acc_length += segment_length;

            if length <= acc_length + T::default_epsilon() {
                let p = compute_bezier_segment_parameter_at_length(
                    &segment,
                    l,
                    tolerance,
                    segment_length,
                    &gauss,
                );
                return Ok(p);
            }
        }

        Err(anyhow::anyhow!(
            "Failed to compute the parameter at the given length"
        ))
    }

    /// Divide a NURBS curve by a given length
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.,
    /// ).unwrap();
    /// let u = std::f64::consts::FRAC_PI_2; // 90 degrees
    /// let params = unit_circle.try_divide_by_length(u).unwrap();
    /// let total_length = 2.0 * std::f64::consts::PI; // circumference of the unit circle
    /// assert_relative_eq!(params[0].length(), 0.);
    /// assert_relative_eq!(params[1].length(), total_length / 4.);
    /// assert_relative_eq!(params[2].length(), total_length / 4. * 2.);
    /// assert_relative_eq!(params[3].length(), total_length / 4. * 3.);
    /// assert_relative_eq!(params[4].length(), total_length);
    /// ```
    pub fn try_divide_by_length(&self, length: T) -> anyhow::Result<Vec<CurveLengthParameter<T>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        anyhow::ensure!(length > T::zero(), "The length must be greater than zero");

        let segments = self.try_decompose_bezier_segments()?;
        let lengths = segments
            .iter()
            .map(|s| s.try_length())
            .collect::<anyhow::Result<Vec<_>>>()?;
        let total = lengths.iter().fold(T::zero(), |a, b| a + *b);

        anyhow::ensure!(
            total > length,
            "The curve is too short to divide by the given length total:{}, given length:{}",
            total,
            length
        );

        let mut samples = vec![CurveLengthParameter::new(self.knots.first(), T::zero())];

        let mut i = 0;
        let mut lc = length;

        let mut acc = T::zero();
        let mut acc_prev = T::zero();

        let gauss = GaussLegendre::new(16 + self.degree)?;
        let eps = T::from_f64(1e-6).unwrap();
        let tolerance = T::from_f64(1e-3 * 2.5).unwrap();

        while i < segments.len() {
            let current_length = lengths[i];
            acc += current_length;

            while lc < acc + eps {
                let u = compute_bezier_segment_parameter_at_length(
                    &segments[i],
                    lc - acc_prev,
                    tolerance,
                    current_length,
                    &gauss,
                );
                samples.push(CurveLengthParameter::new(u, lc));
                lc += length;
            }

            acc_prev += current_length;
            i += 1;
        }

        Ok(samples)
    }

    /// Divide the curve by a given number of segments
    pub fn try_divide_by_count(
        &self,
        segments: usize,
    ) -> anyhow::Result<Vec<CurveLengthParameter<T>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let length = self.try_length()?;
        let u = length / T::from_usize(segments).unwrap();
        self.try_divide_by_length(u)
    }

    /// Try to create a periodic NURBS curve from a set of points
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    /// ];
    /// let closed = NurbsCurve3D::try_periodic(&points, 2);
    /// assert!(closed.is_ok());
    /// let curve = closed.unwrap();
    ///
    /// let (start, end) = curve.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = curve.point_at(start);
    /// let tail = curve.point_at(end);
    /// assert_relative_eq!(head, tail);
    /// ```
    pub fn try_periodic(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let n = points.len();
        if n < degree + 1 {
            anyhow::bail!("Too few control points for curve");
        }

        let pts = points.iter().cycle().take(n + degree);
        let knots = (0..(n + 1 + degree * 2)).map(|i| T::from_usize(i).unwrap());

        Ok(Self {
            control_points: pts
                .map(|p| {
                    let coords = p.to_homogeneous();
                    OPoint::from_slice(coords.as_slice())
                })
                .collect(),
            degree,
            knots: KnotVector::from_iter(knots),
        })
    }

    /// Try to create an interpolated NURBS curve from a set of points
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 2.0, 0.),
    ///     Point3::new(1.0, 2.5, 0.),
    /// ];
    /// let interpolated = NurbsCurve3D::try_interpolate(&points, 3);
    /// assert!(interpolated.is_ok());
    /// let curve = interpolated.unwrap();
    ///
    /// let (start, end) = curve.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = curve.point_at(start);
    /// assert_relative_eq!(points[0], head);
    ///
    /// let tail = curve.point_at(end);
    /// assert_relative_eq!(points[points.len() - 1], tail);
    /// ```
    pub fn try_interpolate(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
    ) -> anyhow::Result<Self>
    where
        DefaultAllocator: Allocator<D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let (control_points, knots) = try_interpolate_control_points(
            &points
                .iter()
                .map(|p| DVector::from_vec(p.iter().copied().collect()))
                .collect_vec(),
            degree,
            true,
        )?;
        Ok(Self {
            degree,
            control_points: control_points
                .iter()
                .map(|v| OPoint::from_slice(v.as_slice()))
                .collect(),
            knots,
        })
    }

    /// Try to create an periodic interpolated NURBS curve from a set of points
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create periodic interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let closed = NurbsCurve3D::try_periodic_interpolate(&points, 3, KnotStyle::Centripetal).unwrap();
    ///
    /// let (start, end) = closed.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = closed.point_at(start);
    /// assert_relative_eq!(points[0], head, epsilon = 1e-10);
    ///
    /// let tail = closed.point_at(end);
    /// assert_relative_eq!(points[0], tail, epsilon = 1e-10);
    /// ```
    pub fn try_periodic_interpolate(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
        knot_style: KnotStyle,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let input = points
            .iter()
            .map(|p| DVector::from_vec(p.iter().copied().collect()))
            .collect_vec();
        let (pts, knots) =
            try_periodic_interpolate_control_points(&input, degree, knot_style, true)?;

        Self::try_new(
            degree,
            pts.iter()
                .map(|v| OPoint::from_slice(v.as_slice()))
                .collect(),
            knots.to_vec(),
        )
    }

    /// Try to create a circle curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.,
    /// ).unwrap();
    /// let (start, end) = unit_circle.knots_domain();
    /// assert_eq!(start, 0.);
    /// assert_eq!(end, std::f64::consts::TAU);
    /// assert_relative_eq!(unit_circle.point_at(start), Point2::new(1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(unit_circle.point_at(end), Point2::new(1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(unit_circle.point_at((start + end) / 2.), Point2::new(-1., 0.), epsilon = 1e-10);
    /// ```
    pub fn try_circle(
        center: &OPoint<T, DimNameDiff<D, U1>>,
        x_axis: &OVector<T, DimNameDiff<D, U1>>,
        y_axis: &OVector<T, DimNameDiff<D, U1>>,
        radius: T,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        Self::try_arc(
            center,
            x_axis,
            y_axis,
            radius,
            T::zero(),
            T::from_f64(TAU).unwrap(),
        )
    }

    /// Try to create an ellipse curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let ellipse = NurbsCurve2D::try_ellipse(
    ///     &Point2::origin(),
    ///     &(Vector2::x() * 2.),
    ///     &(Vector2::y() * 1.25),
    /// ).unwrap();
    /// let (start, end) = ellipse.knots_domain();
    /// assert_eq!(start, 0.);
    /// assert_eq!(end, std::f64::consts::TAU);
    /// assert_relative_eq!(ellipse.point_at(start), Point2::new(2., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(ellipse.point_at(end), Point2::new(2., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(ellipse.point_at((start + end) / 2.), Point2::new(-2., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(ellipse.point_at(std::f64::consts::FRAC_PI_2), Point2::new(0., 1.25), epsilon = 1e-10);
    /// ```
    pub fn try_ellipse(
        center: &OPoint<T, DimNameDiff<D, U1>>,
        x_axis: &OVector<T, DimNameDiff<D, U1>>,
        y_axis: &OVector<T, DimNameDiff<D, U1>>,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        Self::try_ellipse_arc(center, x_axis, y_axis, T::zero(), T::from_f64(TAU).unwrap())
    }

    /// Try to create an arc curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let arc = NurbsCurve2D::try_arc(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.,
    ///     0.,
    ///     std::f64::consts::FRAC_PI_2,
    /// ).unwrap();
    /// let (start, end) = arc.knots_domain();
    /// assert_eq!(start, 0.);
    /// assert_eq!(end, std::f64::consts::FRAC_PI_2);
    /// assert_relative_eq!(arc.point_at(start), Point2::new(1., 0.), epsilon = 1e-10);
    /// let mid = (start + end) / 2.;
    /// let mid_point = mid.cos() * Vector2::x() + mid.sin() * Vector2::y();
    /// assert_relative_eq!(arc.point_at(mid), Point2::from(mid_point), epsilon = 1e-10);
    /// assert_relative_eq!(arc.point_at(end), Point2::new(0., 1.), epsilon = 1e-10);
    /// assert_relative_eq!(arc.tangent_at(start).normalize(), Vector2::new(0., 1.), epsilon = 1e-10);
    /// assert_relative_eq!(arc.tangent_at(end).normalize(), Vector2::new(-1., 0.), epsilon = 1e-10);
    /// ```
    pub fn try_arc(
        center: &OPoint<T, DimNameDiff<D, U1>>,
        x_axis: &OVector<T, DimNameDiff<D, U1>>,
        y_axis: &OVector<T, DimNameDiff<D, U1>>,
        radius: T,
        start_angle: T,
        end_angle: T,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let x_axis = x_axis.normalize() * radius;
        let y_axis = y_axis.normalize() * radius;
        Self::try_ellipse_arc(center, &x_axis, &y_axis, start_angle, end_angle)
    }

    /// Try to create an ellipse arc curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use approx::assert_relative_eq;
    /// let ellipse_arc = NurbsCurve2D::try_ellipse_arc(
    ///     &Point2::origin(),
    ///     &(Vector2::x() * 2.),
    ///     &Vector2::y(),
    ///     0.,
    ///     std::f64::consts::FRAC_PI_2,
    /// ).unwrap();
    /// let (start, end) = ellipse_arc.knots_domain();
    /// assert_eq!(start, 0.);
    /// assert_eq!(end, std::f64::consts::FRAC_PI_2);
    /// assert_relative_eq!(ellipse_arc.point_at(start), Point2::new(2., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(ellipse_arc.point_at(end), Point2::new(0., 1.), epsilon = 1e-10);
    /// ```
    pub fn try_ellipse_arc(
        center: &OPoint<T, DimNameDiff<D, U1>>,
        x_axis: &OVector<T, DimNameDiff<D, U1>>,
        y_axis: &OVector<T, DimNameDiff<D, U1>>,
        start_angle: T,
        end_angle: T,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let x_radius = x_axis.norm();
        let y_radius = y_axis.norm();

        let x_axis = x_axis.normalize();
        let y_axis = y_axis.normalize();

        anyhow::ensure!(
            end_angle > start_angle,
            "`end_angle` must be greater than `start_angle`. {} <= {}",
            end_angle,
            start_angle
        );

        let theta = end_angle - start_angle;

        let arcs = (theta / T::from_f64(FRAC_PI_2).unwrap()).floor().to_usize();
        let arcs = arcs.unwrap_or(1).clamp(1, 4);
        let dtheta = theta / T::from_usize(arcs).unwrap();
        let w1 = (dtheta / T::from_f64(2.0).unwrap()).cos();
        let mut p0 = center
            + &x_axis * x_radius * start_angle.cos()
            + &y_axis * y_radius * start_angle.sin();
        let mut t0 = &y_axis * start_angle.cos() - &x_axis * start_angle.sin();

        let n = 2 * arcs + 1;
        let degree = 2;
        let mut control_points = vec![p0.clone(); n];
        let mut weights = vec![T::one(); n];
        let mut knots = vec![T::zero(); n + degree + 1];

        let mut angle = start_angle;
        let mut index = 0;
        for i in 1..(arcs + 1) {
            angle += dtheta;

            let p2 = center + &x_axis * x_radius * angle.cos() + &y_axis * y_radius * angle.sin();

            let t2 = &y_axis * angle.cos() - &x_axis * angle.sin();
            let ray = Ray::new(p0.clone(), t0.normalize());
            let other = Ray::new(p2.clone(), t2.normalize());
            let intersection = ray
                .find_intersection(&other)
                .ok_or(anyhow::anyhow!("No intersection"))?;

            let p1 = &p0 + &t0 * intersection.intersection0.1;

            control_points[index + 2] = p2.clone();
            weights[index + 2] = T::one();

            control_points[index + 1] = p1;
            weights[index + 1] = w1;

            index += 2;

            if i < arcs {
                p0 = p2;
                t0 = t2;
            }
        }

        for i in 0..3 {
            knots[n + i] = T::one();
        }

        match arcs {
            2 => {
                let v = T::from_f64(0.5).unwrap();
                knots[3] = v;
                knots[4] = v;
            }
            3 => {
                let v1 = T::from_f64(1. / 3.).unwrap();
                let v2 = T::from_f64(2. / 3.).unwrap();
                knots[3] = v1;
                knots[4] = v1;
                knots[5] = v2;
                knots[6] = v2;
            }
            4 => {
                let v1 = T::from_f64(1. / 4.).unwrap();
                let v2 = T::from_f64(2. / 4.).unwrap();
                let v3 = T::from_f64(3. / 4.).unwrap();
                knots[3] = v1;
                knots[4] = v1;
                knots[5] = v2;
                knots[6] = v2;
                knots[7] = v3;
                knots[8] = v3;
            }
            _ => {}
        };

        // convert normalized knot range to the given start and end angle
        knots.iter_mut().for_each(|v| {
            *v = theta * *v + start_angle;
        });

        Ok(Self {
            degree,
            control_points: control_points
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    let mut coords = vec![];
                    let w = weights[i];
                    for i in 0..(D::dim() - 1) {
                        coords.push(p[i] * w);
                    }
                    coords.push(w);
                    OPoint::from_slice(&coords)
                })
                .collect(),
            knots: KnotVector::new(knots),
        })
    }

    /// Try to elevate the degree of the curve
    pub fn try_elevate_degree(&self, target_degree: usize) -> anyhow::Result<Self> {
        if target_degree <= self.degree {
            return Ok(self.clone());
        }

        let n = self.knots.len() - self.degree - 2;
        let new_degree = self.degree;
        let knots = &self.knots;
        let control_points = &self.control_points;
        let degree_inc = target_degree - self.degree;

        //intermediate values
        let mut bezalfs = vec![vec![T::zero(); new_degree + 1]; new_degree + degree_inc + 1];
        let new_control_point_count = control_points.len() + degree_inc + 3;
        let mut bpts = vec![OPoint::origin(); new_control_point_count];
        let mut e_bpts = vec![OPoint::origin(); new_control_point_count];
        let mut next_bpts = vec![OPoint::origin(); new_control_point_count];

        let m = n + new_degree + 1;
        let ph = target_degree;
        let ph2: usize = (T::from_usize(ph).unwrap() / T::from_f64(2.0).unwrap())
            .floor()
            .to_usize()
            .unwrap();

        let mut q_w = vec![OPoint::origin(); new_control_point_count];
        let mut u_h = vec![T::zero(); q_w.len() + target_degree + 1];

        bezalfs[0][0] = T::one();
        bezalfs[ph][new_degree] = T::one();

        let mut binom = Binomial::new();

        for i in 1..=ph2 {
            let inv = T::one() / binom.get(ph, i);
            let mpi = new_degree.min(i);
            for j in 0.max(i - degree_inc)..=mpi {
                bezalfs[i][j] = inv * binom.get(new_degree, j) * binom.get(degree_inc, i - j);
            }
        }

        for i in (ph2 + 1)..ph {
            let mpi = new_degree.min(i);
            for j in 0.max(i - degree_inc)..=mpi {
                bezalfs[i][j] = bezalfs[ph - i][new_degree - j];
            }
        }

        let mh = ph;
        let mut kind = ph + 1;
        let mut r: isize = -1;
        let mut a = new_degree;
        let mut b = new_degree + 1;
        let mut cind = 1;
        let mut ua = knots[0];
        q_w[0] = control_points[0].clone();
        for i in 0..=ph {
            u_h[i] = ua;
        }

        bpts[..(new_degree + 1)].clone_from_slice(&control_points[..(new_degree + 1)]);

        while b < m {
            let i = b;
            while b < m && knots[b] == knots[b + 1] {
                b += 1;
            }
            let mul = b - i + 1;
            let _mh = mh + mul + degree_inc;
            let ub = knots[b];
            let oldr = r;
            r = new_degree as isize - mul as isize;
            let lbz = if oldr > 0 {
                (T::from_isize(oldr + 2).unwrap() / T::from_f64(2.).unwrap())
                    .floor()
                    .to_usize()
                    .unwrap()
            } else {
                1
            };
            let rbz = if r > 0 {
                (T::from_usize(ph).unwrap()
                    - T::from_isize(r + 1).unwrap() / T::from_f64(2.).unwrap())
                .floor()
                .to_usize()
                .unwrap()
            } else {
                ph
            };
            if r > 0 {
                let number = ub - ua;
                let mut alfs = vec![T::zero(); new_degree];
                let mut k = new_degree;
                while k > mul {
                    alfs[k - mul - 1] = number / (knots[a + k] - ua);
                    k -= 1;
                }
                for j in 1..=(r as usize) {
                    let save = (r as usize) - j;
                    let s = mul + j;
                    let mut k = new_degree;
                    while k >= s {
                        bpts[k] = bpts[k].lerp(&bpts[k - 1], T::one() - alfs[k - s]);
                        k -= 1;
                    }
                    next_bpts[save] = bpts[new_degree].clone();
                }
            }

            for i in lbz..=ph {
                e_bpts[i] = OPoint::origin();
                let mpi = new_degree.min(i);
                for j in 0.max(i - degree_inc)..=mpi {
                    e_bpts[i].coords = &e_bpts[i].coords + &bpts[j].coords * bezalfs[i][j];
                }
            }

            if oldr > 1 {
                let mut first = kind - 2;
                let mut last = kind;
                let den = ub - ua;
                let bet = (ub - u_h[kind - 1]) / den;
                for tr in 1..oldr {
                    let mut i = first;
                    let mut j = last;
                    let mut kj = j - kind + 1;
                    let utr = tr as usize;
                    while (j as isize - i as isize) > tr {
                        if i < cind {
                            let alf = (ub - u_h[i]) / (ua - u_h[i]);
                            q_w[i] = q_w[i].lerp(&q_w[i - 1], T::one() - alf);
                        }
                        if j >= lbz {
                            if (j as isize) - tr <= (kind as isize - ph as isize + oldr) {
                                let gam = (ub - u_h[j - utr]) / den;
                                e_bpts[kj] = e_bpts[kj].lerp(&e_bpts[kj + 1], T::one() - gam);
                            }
                        } else {
                            e_bpts[kj] = e_bpts[kj].lerp(&e_bpts[kj + 1], T::one() - bet);
                        }
                        i += 1;
                        j -= 1;
                        kj -= 1;
                    }
                    first -= 1;
                    last += 1;
                }
            }

            if a != new_degree {
                for _i in 0..(ph as isize - oldr) {
                    u_h[kind] = ua;
                    kind += 1;
                }
            }

            for j in lbz..=rbz {
                q_w[cind] = e_bpts[j].clone();
                cind += 1;
            }

            if b < m {
                let ur = r as usize;
                bpts[..ur].clone_from_slice(&next_bpts[..ur]);
                for j in (r as usize)..=new_degree {
                    bpts[j] = control_points[b - new_degree + j].clone();
                }
                a = b;
                b += 1;
                ua = ub;
            } else {
                for i in 0..=ph {
                    u_h[kind + i] = ub;
                }
            }
        }

        Ok(Self {
            degree: target_degree,
            control_points: q_w,
            knots: KnotVector::new(u_h),
        })
    }

    /// Try to add a knot to the curve
    pub fn try_add_knot(&mut self, knot: T) -> anyhow::Result<()> {
        anyhow::ensure!(
            knot >= self.knots[0],
            "Knot is smaller than the first knot: {} < {}",
            knot,
            self.knots[0]
        );
        anyhow::ensure!(
            knot <= self.knots[self.knots.len() - 1],
            "Knot is larger than the last knot: {} > {}",
            knot,
            self.knots[self.knots.len() - 1]
        );

        let k = self.degree;
        let n = self.control_points.len();
        let idx = self.knots.add(knot);
        let start = idx.saturating_sub(k);
        let end = if idx > n {
            self.control_points
                .push(self.control_points.last().unwrap().clone());
            n + 1
        } else {
            self.control_points
                .insert(idx - 1, self.control_points[idx - 1].clone());
            idx
        };

        for i in start..end {
            let i0 = end + start - i - 1;
            let delta = self.knots[i0 + k + 1] - self.knots[i0];
            let inv = if delta != T::zero() {
                T::one() / delta
            } else {
                T::zero()
            };
            let a = (self.knots[idx] - self.knots[i0]) * inv;
            let delta_control_point = if i0 == 0 {
                self.control_points[i0].coords.clone()
            } else if i0 == self.control_points.len() {
                -self.control_points[i0 - 1].coords.clone()
            } else {
                &self.control_points[i0] - &self.control_points[i0 - 1]
            };
            let mut p = delta_control_point * (T::one() - a);
            p[D::dim() - 1] = T::zero();
            self.control_points[i0].coords -= p;
        }

        Ok(())
    }

    /// Try to remove a knot from the curve
    /// Returns the number of knots removed
    /// `tolerance` defines the acceptable deviation in the curve's shape.
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// let mut polyline = NurbsCurve3D::polyline(
    ///     &vec![
    ///         Point3::new(0., 0., 0.),
    ///         Point3::new(1., 0., 0.),
    ///         Point3::new(2., 0., 0.),
    ///         Point3::new(3., 1., 0.),
    ///         Point3::new(4., 2., 0.),
    ///     ],
    ///     false,
    /// );
    /// assert_eq!(polyline.control_points().len(), 5);
    /// assert_eq!(polyline.knots().len(), 7);
    ///
    /// let res = polyline.try_remove_knot(1.0, None).unwrap();
    /// assert_eq!(res, 1);
    /// assert_eq!(polyline.control_points().len(), 4);
    /// assert_eq!(polyline.knots().len(), 6);
    /// ```
    pub fn try_remove_knot(&mut self, knot: T, tolerance: Option<T>) -> anyhow::Result<usize> {
        let tolerance = tolerance.unwrap_or(T::from_f64(1e-6).unwrap());

        let n = self.control_points.len() - 1;
        let m = n + self.degree + 1;

        let r = self
            .knots
            .iter()
            .position(|k| *k == knot)
            .ok_or(anyhow::anyhow!("Knot not found"))?; // index of the knot

        let multiplicities = self.knots.multiplicity();
        let mult = multiplicities
            .into_iter()
            .find(|m| *m.knot() == knot)
            .ok_or(anyhow::anyhow!("Knot not found"))?;
        let s = mult.multiplicity();
        let p = self.degree();
        let ord = p + 1;

        if r <= p {
            return Err(anyhow::anyhow!(
                "Knot is too close to the start of the curve"
            ));
        }
        let mut first = r - p;

        if r <= s {
            return Err(anyhow::anyhow!("Knot is not removable"));
        }
        let mut last = r - s;

        let mut removed_knots = self.knots.to_vec();
        let mut removed_control_points = self.control_points.to_vec();
        let mut temp = vec![OPoint::<T, D>::origin(); 2 * (m + 1)];

        let mut t = 0usize;

        // use multiplicity as # of times to remove knot
        for _ in 0..s {
            let off = first - 1;
            temp[0] = removed_control_points[off].clone();

            let idx_temp_right = last + 1 - off;
            let idx_pw_right = (last + 1) as usize;
            temp[idx_temp_right] = removed_control_points[idx_pw_right].clone();

            let mut i = first;
            let mut j = last;
            let mut ii = 1usize;
            let mut jj = idx_temp_right - 1;

            while j >= i && (j - i) > t {
                // compute new control points for one removal step
                let alfi =
                    (knot - removed_knots[i]) / (removed_knots[i + ord + t] - removed_knots[i]);
                let alfj =
                    (knot - removed_knots[j - t]) / (removed_knots[j + ord] - removed_knots[j - t]);

                let left = &temp[ii - 1];
                let right = &temp[jj + 1];

                let temp_ii =
                    ((&removed_control_points[i].coords - &left.coords * (T::one() - alfi)) / alfi)
                        .into();

                let temp_jj = ((&removed_control_points[j].coords - &right.coords * alfj)
                    / (T::one() - alfj))
                    .into();

                temp[ii] = temp_ii;
                temp[jj] = temp_jj;

                i += 1;
                j -= 1;
                ii += 1;
                jj -= 1;
            }

            // check if knot removable
            let remove = if j >= i && (j - i) < t {
                let left = &temp[ii - 1];
                let right = &temp[jj + 1];
                (left - right).norm() <= tolerance
            } else {
                let alfi =
                    (knot - removed_knots[i]) / (removed_knots[i + ord + t] - removed_knots[i]);

                let pwi = &removed_control_points[i];
                let lo = &temp[ii - 1];
                let hi = &temp[ii + t + 1];

                let interpolated = &hi.coords * alfi + &lo.coords * (T::one() - alfi);
                let delta = &pwi.coords - interpolated;
                delta.norm() <= tolerance
            };

            if !remove {
                break;
            }

            let mut i2 = first;
            let mut j2 = last;
            while j2 >= i2 && (j2 - i2) > t {
                removed_control_points[i2] = temp[i2 - off].clone();
                removed_control_points[j2] = temp[j2 - off].clone();

                i2 += 1;
                j2 -= 1;
            }

            // expand first and last
            first -= 1;
            last += 1;

            t += 1;
        }

        // shift knots and control points to the front
        if t > 0 {
            // shift knots
            for k in (r + 1)..=m {
                removed_knots[k - t] = removed_knots[k];
            }

            // truncate knots
            removed_knots.truncate(m + 1 - t);

            // shift control points
            let fout = (2 * r - s - p) / 2;
            let mut j = fout;
            let mut i = j;
            for k in 1..t {
                if k % 2 == 1 {
                    i += 1;
                } else {
                    j = j.saturating_sub(1);
                }
            }
            for k in (i + 1)..=n {
                removed_control_points[j] = removed_control_points[k].clone();
                j += 1;
            }

            // truncate control points
            removed_control_points.truncate(n + 1 - t);

            self.knots = KnotVector::new(removed_knots);
            self.control_points = removed_control_points;
        }

        Ok(t)
    }

    /// Check if the curve is clamped
    pub fn is_clamped(&self) -> bool {
        self.knots.is_clamped(self.degree)
    }

    /// Check if the curve is closed
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Point3, Vector3};
    /// let circle = NurbsCurve3D::try_circle(
    ///     &Point3::origin(),
    ///     &Vector3::x(),
    ///     &Vector3::y(),
    ///    1.,
    /// ).unwrap();
    /// assert!(circle.is_closed());
    ///
    /// let polyline = NurbsCurve3D::polyline(
    ///     &vec![
    ///         Point3::new(-1., -1., 0.),
    ///         Point3::new(1., -1., 0.),
    ///         Point3::new(1., 1., 0.),
    ///         Point3::new(-1., 1., 0.),
    ///         Point3::new(-1., -1., 0.),
    ///     ], true,
    /// );
    /// assert!(polyline.is_closed());
    ///
    /// let unclosed = NurbsCurve3D::polyline(
    ///     &vec![
    ///         Point3::new(-1., -1., 0.),
    ///         Point3::new(1., -1., 0.),
    ///         Point3::new(1., 1., 0.),
    ///         Point3::new(-1., 1., 0.),
    ///     ], true,
    /// );
    /// assert!(!unclosed.is_closed());
    /// ```
    pub fn is_closed(&self) -> bool
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let eps = T::default_epsilon() * T::from_usize(10).unwrap();
        match self.knots.is_clamped(self.degree) {
            true => {
                let pts = self.dehomogenized_control_points();
                let delta = &pts[0] - &pts[self.control_points.len() - 1];
                delta.norm() < eps
            }
            false => {
                let (s, e) = self.knots_domain();
                let s = self.point_at(s);
                let e = self.point_at(e);
                let delta = s - e;
                delta.norm() < eps
            }
        }
    }

    /// Try to refine the curve by inserting knots
    pub fn try_refine_knot(&mut self, knots_to_insert: Vec<T>) -> anyhow::Result<()> {
        if knots_to_insert.is_empty() {
            return Ok(());
        }

        let degree = self.degree;
        let control_points = &self.control_points;

        let n = control_points.len() - 1;
        let m = n + degree + 1;
        let r = knots_to_insert.len() - 1;
        let a = self
            .knots
            .find_knot_span_index(n, degree, knots_to_insert[0]);
        let b = self
            .knots
            .find_knot_span_index(n, degree, knots_to_insert[r])
            + 1;

        let mut control_points_post = vec![OPoint::<T, D>::origin(); n + r + 2];
        let mut knots_post = vec![T::zero(); m + 1 + r + 1];
        // assert!(knots_post.len() == control_points_post.len() + degree + 1);

        control_points_post[..((a - degree) + 1)]
            .clone_from_slice(&control_points[..((a - degree) + 1)]);
        for i in (b - 1)..=n {
            control_points_post[i + r + 1] = control_points[i].clone();
        }

        for i in 0..=a {
            knots_post[i] = self.knots[i];
        }
        for i in (b + degree)..=m {
            knots_post[i + r + 1] = self.knots[i];
        }

        let mut i = b + degree - 1;
        let mut k = b + degree + r;

        for j in (0..=r).rev() {
            while knots_to_insert[j] <= self.knots[i] && i > a {
                control_points_post[k - degree - 1] = control_points[i - degree - 1].clone();
                knots_post[k] = self.knots[i];
                k -= 1;
                i -= 1;
            }
            control_points_post[k - degree - 1] = control_points_post[k - degree].clone();
            for l in 1..=degree {
                let ind = k - degree + l;
                if ind < control_points_post.len() {
                    let alpha = knots_post[k + l] - knots_to_insert[j];
                    if alpha.abs() < T::default_epsilon() {
                        control_points_post[ind - 1] = control_points_post[ind].clone();
                    } else {
                        let denom = knots_post[k + l] - self.knots[i - degree + l];
                        let weight = if denom != T::zero() {
                            alpha / denom
                        } else {
                            T::zero()
                        };
                        control_points_post[ind - 1] = control_points_post[ind - 1]
                            .lerp(&control_points_post[ind], T::one() - weight);
                    }
                } else {
                    // TODO: resolve this issue
                    // ind is out of bound
                }
            }
            knots_post[k] = knots_to_insert[j];
            k -= 1;
        }

        self.knots = KnotVector::new(knots_post);
        self.control_points = control_points_post;

        Ok(())
    }

    /// Find the closest point on the curve to a given point
    pub fn find_closest_point(
        &self,
        point: &OPoint<T, DimNameDiff<D, U1>>,
    ) -> anyhow::Result<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        self.find_closest_parameter(point).map(|u| self.point_at(u))
    }

    /// Find the closest parameter on the curve to a given point with Newton's method
    pub fn find_closest_parameter(&self, point: &OPoint<T, DimNameDiff<D, U1>>) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let (min_u, max_u) = self.knots_domain();
        let samples = self.control_points.len() * self.degree;
        let pts = self.sample_regular_range_with_parameter(min_u, max_u, samples);

        let mut min = <T as RealField>::max_value().unwrap();
        let mut u = min_u;

        let closed = self.is_closed();

        for i in 0..pts.len() - 1 {
            let u0 = pts[i].0;
            let u1 = pts[i + 1].0;

            let p0 = &pts[i].1;
            let p1 = &pts[i + 1].1;

            let (proj_u, proj_pt) = segment_closest_point(point, p0, p1, u0, u1);
            let d = (point - proj_pt).norm();

            if d < min {
                min = d;
                u = proj_u;
            }
        }

        let solver = CurveClosestParameterNewton::new((min_u, max_u), closed);
        let res = Executor::new(CurveClosestParameterProblem::new(point, self), solver)
            .configure(|state| state.param(u).max_iters(5))
            .run()?;
        match res.state().get_best_param().cloned() {
            Some(t) => {
                if t.is_finite() {
                    Ok(t)
                } else {
                    Err(anyhow::anyhow!("No best parameter found"))
                }
            }
            _ => Err(anyhow::anyhow!("No best parameter found")),
        }
    }

    /// Try to clamp knots of the curve
    /// Multiplex the start/end part of the knot vector so that the knot has `degree + 1` overlap
    pub fn try_clamp(&mut self) -> anyhow::Result<()> {
        let degree = self.degree();

        let start = self.knots.first();
        let end = self.knots.last();
        let multiplicity = self.knots.multiplicity();
        let start_knot_count = multiplicity
            .iter()
            .find(|m| *m.knot() == start)
            .ok_or(anyhow::anyhow!("Start knot not found"))?
            .multiplicity();
        let end_knot_count = multiplicity
            .iter()
            .find(|m| *m.knot() == end)
            .ok_or(anyhow::anyhow!("End knot not found"))?
            .multiplicity();

        for _ in start_knot_count..=degree {
            self.try_add_knot(start)?;
        }
        for _ in end_knot_count..=degree {
            self.try_add_knot(end)?;
        }

        Ok(())
    }

    /// Decompose the curve into Bezier segments
    pub fn try_decompose_bezier_segments(&self) -> anyhow::Result<Vec<Self>> {
        let mut cloned = self.clone();
        if !cloned.is_clamped() {
            cloned.try_clamp()?;
        }

        let knot_mults = cloned.knots.multiplicity();
        let req_mult = cloned.degree + 1;

        for knot_mult in knot_mults.iter() {
            if knot_mult.multiplicity() < req_mult {
                let knots_insert = vec![*knot_mult.knot(); req_mult - knot_mult.multiplicity()];
                cloned.try_refine_knot(knots_insert)?;
            }
        }

        let div = cloned.knots().len() / req_mult - 1;
        let knot_length = req_mult * 2;
        let mut segments = vec![];

        for i in 0..div {
            let start = i * req_mult;
            let end = start + knot_length;
            let knots = cloned.knots().as_slice()[start..end].to_vec();
            let control_points = cloned.control_points[start..(start + req_mult)].to_vec();
            segments.push(Self {
                degree: self.degree,
                control_points,
                knots: KnotVector::new(knots),
            });
        }

        if segments.len() <= 1 {
            return Ok(vec![cloned]);
        }

        Ok(segments)
    }

    /// Cast the curve to a curve with another floating point type
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> NurbsCurve<F, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        NurbsCurve {
            control_points: self
                .control_points
                .iter()
                .map(|p| p.clone().cast())
                .collect(),
            degree: self.degree,
            knots: self.knots.cast(),
        }
    }

    /// Try to reduce knots of the curve as much as possible
    /// Returns the total number of knots removed
    /// `tolerance` defines the acceptable deviation in the curve's shape.
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    ///
    /// // Create a curve with redundant knots
    /// let points = vec![
    ///     Point3::new(0., 0., 0.),
    ///     Point3::new(1., 0., 0.),
    ///     Point3::new(2., 0., 0.),
    ///     Point3::new(3., 1., 0.),
    ///     Point3::new(4., 2., 0.),
    /// ];
    /// let mut curve = NurbsCurve3D::polyline(&points, false);
    ///
    /// // Add some redundant knots
    /// curve.try_add_knot(0.5).unwrap();
    /// curve.try_add_knot(1.5).unwrap();
    /// curve.try_add_knot(2.5).unwrap();
    ///
    /// let original_knot_count = curve.knots().len();
    /// let original_control_point_count = curve.control_points().len();
    ///
    /// // Reduce knots
    /// let removed_count = curve.try_reduce_knots(Some(1e-6)).unwrap();
    ///
    /// let final_knot_count = curve.knots().len();
    /// let final_control_point_count = curve.control_points().len();
    ///
    /// // Verify that knots and control points were reduced
    /// assert!(final_knot_count < original_knot_count);
    /// assert!(final_control_point_count <= original_control_point_count);
    /// assert!(removed_count > 0);
    /// ```
    pub fn try_reduce_knots(&mut self, tolerance: Option<T>) -> anyhow::Result<usize> {
        let tolerance = tolerance.unwrap_or(T::from_f64(1e-6).unwrap());
        let mut total_removed = 0;
        let mut changed = true;

        // Continue until no more knots can be removed
        while changed {
            changed = false;
            let multiplicities = self.knots.multiplicity();

            // Try to remove knots starting from those with highest multiplicity
            // and avoid the first and last knots (domain boundaries)
            let mut candidates: Vec<_> = multiplicities
                .iter()
                .filter(|m| {
                    let knot = *m.knot();
                    let first_knot = self.knots.first();
                    let last_knot = self.knots.last();
                    // Skip boundary knots and knots with multiplicity 0
                    knot != first_knot && knot != last_knot && m.multiplicity() > 0
                })
                .collect();

            // Sort by multiplicity (descending) to prioritize removing high-multiplicity knots
            candidates.sort_by_key(|m| std::cmp::Reverse(m.multiplicity()));

            for mult_info in candidates {
                let knot = *mult_info.knot();

                // Try to remove this knot as many times as possible
                let mut local_removed = 0;
                loop {
                    match self.try_remove_knot(knot, Some(tolerance)) {
                        Ok(removed) if removed > 0 => {
                            local_removed += removed;
                            changed = true;
                        }
                        _ => break, // Can't remove this knot anymore
                    }
                }

                total_removed += local_removed;
            }
        }

        Ok(total_removed)
    }
}

/// Enable to transform a NURBS curve by a given DxD matrix
impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for NurbsCurve<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.control_points.iter_mut().for_each(|p| {
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
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Reverse the direction of the curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// use approx::assert_relative_eq;
    /// let points = vec![
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 0.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(0.0, 1.0),
    /// ];
    /// let mut curve = NurbsCurve2D::try_interpolate(&points, 3).unwrap();
    /// curve.invert();
    /// let (start, end) = curve.knots_domain();
    /// assert_relative_eq!(curve.point_at(start), points[points.len() - 1]);
    /// assert_relative_eq!(curve.point_at(end), points[0]);
    /// ```
    fn invert(&mut self) {
        self.control_points.reverse();
        self.knots.invert();
    }
}

impl<T: FloatingPoint> NurbsCurve3D<T> {
    /// Compute the Frenet frames of the curve at given parameters
    /// based on the method described in the paper: http://www.cs.indiana.edu/pub/techreports/TR425.pdf
    pub fn compute_frenet_frames(&self, parameters: &[T]) -> Vec<FrenetFrame<T>> {
        let tangents: Vec<_> = parameters
            .iter()
            .map(|u| self.tangent_at(*u).normalize())
            .collect();
        let mut normals = vec![Vector3::zeros()];
        let mut binormals = vec![Vector3::zeros()];

        let mut normal = Vector3::zeros();
        let tx = tangents[0].x.abs();
        let ty = tangents[0].y.abs();
        let tz = tangents[0].z.abs();

        let mut min = T::max_value().unwrap();
        if tx <= min {
            min = tx;
            normal = Vector3::x();
        }
        if ty <= min {
            min = ty;
            normal = Vector3::y();
        }
        if tz <= min {
            normal = Vector3::z();
        }

        let v = tangents[0].cross(&normal).normalize();
        normals[0] = tangents[0].cross(&v).normalize();
        binormals[0] = tangents[0].cross(&normals[0]).normalize();

        for i in 1..parameters.len() {
            let prev_normal = &normals[i - 1];

            let v = tangents[i - 1].cross(&tangents[i]).normalize();
            if v.norm() > T::default_epsilon() {
                let theta = tangents[i - 1]
                    .dot(&tangents[i])
                    .clamp(-T::one(), T::one())
                    .acos();
                let rot = Rotation3::from_axis_angle(&UnitVector3::new_normalize(v), theta);
                normals.push(rot * prev_normal);
            } else {
                normals.push(*prev_normal);
            }

            binormals.push(tangents[i].cross(&normals[i]).normalize());
        }

        parameters
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let position = self.point_at(*t);
                FrenetFrame::new(position, tangents[i], normals[i], binormals[i])
            })
            .collect()
    }
}

/// Find the curve parameter at arc length on a Bezier segment of a NURBS curve
/// by binary search
fn compute_bezier_segment_parameter_at_length<T: FloatingPoint, D>(
    s: &NurbsCurve<T, D>,
    length: T,
    tolerance: T,
    total_length: T,
    gauss: &GaussLegendre,
) -> T
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let (k0, k1) = s.knots_domain();
    if length <= T::zero() {
        return k0;
    } else if total_length <= length {
        return k1;
    }

    let mut start = (k0, T::zero());
    let mut end = (k1, total_length);

    let inv = T::one() / T::from_usize(2).unwrap();

    // binary search
    while (end.1 - start.1) > tolerance {
        let middle_parameter = (start.0 + end.0) * inv;
        let mid = (
            middle_parameter,
            compute_bezier_segment_length(s, middle_parameter, gauss),
        );
        if mid.1 > length {
            end = mid;
        } else {
            start = mid;
        }
    }

    (start.0 + end.0) * inv
}

/// Compute the length of a Bezier segment of a NURBS curve
/// by gauss-legendre quadrature
fn compute_bezier_segment_length<T: FloatingPoint, D>(
    s: &NurbsCurve<T, D>,
    u: T,
    gauss: &GaussLegendre,
) -> T
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let (start, end) = s.knots_domain();
    if start + T::default_epsilon() < u {
        let t = end.min(u);
        let left = start.to_f64().unwrap();
        let right = t.to_f64().unwrap();
        let sum = gauss.integrate(left, right, |x| {
            let x = T::from_f64(x).unwrap();
            let deriv = s.rational_derivatives(x, 1);
            let tan = deriv[1].norm();
            tan.to_f64().unwrap()
        });
        T::from_f64(sum).unwrap()
    } else {
        T::zero()
    }
}

/// Dehomogenize a point
pub fn dehomogenize<T: FloatingPoint, D>(
    point: &OPoint<T, D>,
) -> Option<OPoint<T, DimNameDiff<D, U1>>>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let v = &point.coords;
    let idx = D::dim() - 1;
    let w = v[idx];
    if w != T::zero() {
        let coords =
            v.generic_view((0, 0), (<D as DimNameSub<U1>>::Output::name(), Const::<1>)) / w;
        Some(OPoint { coords })
    } else {
        None
    }
}

pub fn try_interpolate_control_points<T: FloatingPoint>(
    points: &[DVector<T>],
    degree: usize,
    homogeneous: bool,
) -> anyhow::Result<(Vec<DVector<T>>, KnotVector<T>)> {
    let n = points.len();
    if n < degree + 1 {
        anyhow::bail!("Too few control points for curve");
    }

    let mut us: Vec<T> = vec![T::zero()];
    for i in 1..n {
        let sub = &points[i] - &points[i - 1];
        let chord = sub.norm();
        let last = us[i - 1];
        us.push(last + chord);
    }

    // normalize
    let max = us[us.len() - 1];
    for i in 0..us.len() {
        us[i] /= max;
    }

    let mut knots_start = vec![T::zero(); degree + 1];

    let start = 1;
    let end = us.len() - degree;

    for i in start..end {
        let mut weight_sums = T::zero();
        for j in 0..degree {
            weight_sums += us[i + j];
        }
        knots_start.push(weight_sums / T::from_usize(degree).unwrap());
    }

    let knots = KnotVector::new([knots_start, vec![T::one(); degree + 1]].concat());
    let plen = points.len();

    let n = plen - 1;
    let ld = plen - (degree + 1);

    // build basis function coefficients matrix
    let mut m_a = DMatrix::<T>::zeros(us.len(), degree + 1 + ld);

    for i in 0..us.len() {
        let u = us[i];
        let knot_span_index = knots.find_knot_span_index(n, degree, u);
        let basis = knots.basis_functions(knot_span_index, u, degree);

        let ls = knot_span_index - degree;
        let row_start = vec![T::zero(); ls];
        let row_end = vec![T::zero(); ld - ls];
        let e = [row_start, basis, row_end].concat();
        // println!("e = {:?}", &e);
        for j in 0..e.len() {
            m_a[(i, j)] = e[j];
        }
    }

    let control_points = try_solve_interpolation(m_a, points, homogeneous)?;

    Ok((control_points, knots))
}

pub fn try_periodic_interpolate_control_points<T: FloatingPoint>(
    points: &[DVector<T>],
    degree: usize,
    knot_style: KnotStyle,
    homogeneous: bool,
) -> anyhow::Result<(Vec<DVector<T>>, KnotVector<T>)> {
    let n = points.len();
    if n < degree + 1 {
        anyhow::bail!("Too few control points for curve");
    }

    // build periodic knot vector
    let parameters = knot_style.parameterize(points, true);

    let head = &parameters[0..(degree - 1)];
    let tail = &parameters[(parameters.len() - degree + 1)..];
    let start_parameters = tail.to_vec();

    let knots = [start_parameters, parameters.clone(), head.to_vec()].concat();

    let m = points.len() + degree;

    let knots = [
        vec![T::zero()],
        knots
            .iter()
            .scan(T::zero(), |p, x| {
                *p += *x;
                Some(*p)
            })
            .collect_vec(),
    ]
    .concat();

    // translate knot vectors to fit NURBS requirements
    let k0 = if degree > 2 {
        knots[0] - (knots[m + 1] - knots[m])
    } else {
        knots[0]
    };
    let k1 = if degree > 2 {
        knots[knots.len() - 1] + (knots[degree + 1] - knots[degree])
    } else {
        knots[knots.len() - 1]
    };
    let knots = [vec![k0], knots, vec![k1]].concat();

    let knots_vec = KnotVector::new(knots.clone());
    let plen = points.len();

    // build basis function coefficients matrix
    let mut m_a = DMatrix::<T>::zeros(plen, plen);
    let zero_pad = vec![T::zero(); plen - (degree + 1)];

    let n = knots_vec.len() - degree - 2;
    for i in 0..plen {
        let u = knots[i + degree]; // from start domain
        let knot_span_index = knots_vec.find_knot_span_index(n, degree, u);

        let basis = knots_vec.basis_functions(knot_span_index, u, degree);
        let basis_padded = [basis, zero_pad.clone()].concat();

        let ls = knot_span_index - degree;

        // In a closed periodic NURBS curve, the control points loop.
        // The solver only solves for the control points that are not looping,
        // so the coefficient matrix is looped to represent the duplicated control points.
        let n = basis_padded.len() - ls;
        let mut e = basis_padded[n..].to_vec();
        e.extend_from_slice(&basis_padded[0..n]);

        for j in 0..e.len() {
            m_a[(i, j)] = e[j];
        }
    }

    let mut control_points = try_solve_interpolation(m_a, points, homogeneous)?;

    // periodic
    for i in 0..(degree) {
        control_points.push(control_points[i].clone());
    }

    Ok((control_points, knots_vec))
}

fn try_solve_interpolation<T: FloatingPoint>(
    m_a: DMatrix<T>,
    points: &[DVector<T>],
    homogeneous: bool,
) -> anyhow::Result<Vec<DVector<T>>> {
    let n = points.len();
    let dim = points[0].len();

    let lu = m_a.lu();
    let mut m_x = DMatrix::<T>::identity(n, dim);
    let rows = m_x.nrows();
    for i in 0..dim {
        let b: Vec<_> = points.iter().map(|p| p[i]).collect();
        let b = DVector::from_vec(b);
        // println!("b = {:?}", &b);
        let xs = lu.solve(&b).ok_or(anyhow::anyhow!("Solve failed"))?;
        for j in 0..rows {
            m_x[(j, i)] = xs[j];
        }
    }

    // extract control points from solved x
    let mut control_points = vec![];
    for i in 0..m_x.nrows() {
        let mut coords = vec![];
        for j in 0..m_x.ncols() {
            coords.push(m_x[(i, j)]);
        }
        if homogeneous {
            coords.push(T::one());
        }
        control_points.push(DVector::from_vec(coords));
    }

    Ok(control_points)
}

#[cfg(feature = "serde")]
impl<T, D: DimName> serde::Serialize for NurbsCurve<T, D>
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
        let mut state = serializer.serialize_struct("NurbsCurve", 3)?;
        state.serialize_field("control_points", &self.control_points)?;
        state.serialize_field("degree", &self.degree)?;
        state.serialize_field("knots", &self.knots)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, D: DimName> serde::Deserialize<'de> for NurbsCurve<T, D>
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
            Degree,
            Knots,
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
                        formatter.write_str("`control_points` or `degree` or `knots`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "control_points" => Ok(Field::ControlPoints),
                            "degree" => Ok(Field::Degree),
                            "knots" => Ok(Field::Knots),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct NurbsCurveVisitor<T, D>(std::marker::PhantomData<(T, D)>);

        impl<T, D> NurbsCurveVisitor<T, D> {
            pub fn new() -> Self {
                NurbsCurveVisitor(std::marker::PhantomData)
            }
        }

        impl<'de, T, D: DimName> Visitor<'de> for NurbsCurveVisitor<T, D>
        where
            T: FloatingPoint + serde::Deserialize<'de>,
            DefaultAllocator: Allocator<D>,
            <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>:
                serde::Deserialize<'de>,
        {
            type Value = NurbsCurve<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct NurbsCurve")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut control_points = None;
                let mut degree = None;
                let mut knots = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::ControlPoints => {
                            if control_points.is_some() {
                                return Err(de::Error::duplicate_field("control_points"));
                            }
                            control_points = Some(map.next_value()?);
                        }
                        Field::Degree => {
                            if degree.is_some() {
                                return Err(de::Error::duplicate_field("degree"));
                            }
                            degree = Some(map.next_value()?);
                        }
                        Field::Knots => {
                            if knots.is_some() {
                                return Err(de::Error::duplicate_field("knots"));
                            }
                            knots = Some(map.next_value()?);
                        }
                    }
                }
                let control_points =
                    control_points.ok_or_else(|| de::Error::missing_field("control_points"))?;
                let degree = degree.ok_or_else(|| de::Error::missing_field("degree"))?;
                let knots = knots.ok_or_else(|| de::Error::missing_field("knots"))?;

                Ok(Self::Value {
                    control_points,
                    degree,
                    knots,
                })
            }
        }

        const FIELDS: &[&str] = &["control_points", "degree", "knots"];
        deserializer.deserialize_struct("NurbsCurve", FIELDS, NurbsCurveVisitor::<T, D>::new())
    }
}
