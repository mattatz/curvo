use std::vec;

use argmin::core::{ArgminFloat, Executor, State};

use argmin_math::ArgminScaledSub;
use gauss_quad::GaussLegendre;
use nalgebra::allocator::Allocator;
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub,
    DimNameSum, OMatrix, OPoint, OVector, RealField, Rotation3, UnitVector3, Vector3, U1,
};
use rand::rngs::ThreadRng;
use rand::Rng;
use simba::scalar::SupersetOf;

use crate::intersection::curve_intersection::CurveIntersection;
use crate::misc::binomial::Binomial;
use crate::misc::frenet_frame::FrenetFrame;
use crate::misc::transformable::Transformable;
use crate::misc::trigonometry::{segment_closest_point, three_points_are_flat};
use crate::prelude::{BoundingBoxTraversal, CurveLengthParameter, Invertible, KnotVector};
use crate::{misc::FloatingPoint, ClosestParameterNewton, ClosestParameterProblem};

/// NURBS curve representation
/// By generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug)]
pub struct NurbsCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
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
    DefaultAllocator: Allocator<T, D>,
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

    /// Create a dehomogenized version of the curve
    pub fn dehomogenize(&self) -> NurbsCurve<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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

    /// Tessellate the curve using an adaptive algorithm
    /// this `adaptive` means that the curve will be tessellated based on the curvature of the curve
    pub fn tessellate(&self, tolerance: Option<T>) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        if self.degree == 1 {
            return self.dehomogenized_control_points();
        }

        let mut rng = rand::thread_rng();
        let tol = tolerance.unwrap_or(T::from_f64(1e-3).unwrap());
        let (start, end) = self.knots_domain();
        self.tessellate_adaptive(start, end, tol, &mut rng)
    }

    /// Tessellate the curve using an adaptive algorithm recursively
    /// if the curve between [start ~ end] is flat enough, it will return the two end points
    fn tessellate_adaptive(
        &self,
        start: T,
        end: T,
        tol: T,
        rng: &mut ThreadRng,
    ) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let p1 = self.point_at(start);
        let p3 = self.point_at(end);

        let t = 0.5_f64 + 0.2_f64 * rng.gen::<f64>();
        let delta = end - start;
        if delta < T::from_f64(1e-8).unwrap() {
            return vec![p1];
        }

        let mid = start + delta * T::from_f64(t).unwrap();
        let p2 = self.point_at(mid);

        let diff = &p1 - &p3;
        let diff2 = &p1 - &p2;
        if (diff.dot(&diff) < tol && diff2.dot(&diff2) > tol)
            || !three_points_are_flat(&p1, &p2, &p3, tol)
        {
            let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
            let mut left_pts = self.tessellate_adaptive(start, exact_mid, tol, rng);
            let right_pts = self.tessellate_adaptive(exact_mid, end, tol, rng);
            left_pts.pop();
            [left_pts, right_pts].concat()
        } else {
            vec![p1, p3]
        }
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let deriv = self.rational_derivatives(u, 1);
        deriv[1].clone()
    }

    /// Evaluate the rational derivatives at a given parameter
    pub(crate) fn rational_derivatives(
        &self,
        u: T,
        derivs: usize,
    ) -> Vec<OVector<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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

    /// Compute the length of the curve by gauss-legendre quadrature
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    /// let corner_weight = 1. / 2.;
    /// let unit_circle = NurbsCurve2D::try_new(
    ///     2,
    ///     vec![
    ///         Point3::new(1.0, 0.0, 1.),
    ///         Point3::new(1.0, 1.0, 1.0) * corner_weight,
    ///         Point3::new(-1.0, 1.0, 1.0) * corner_weight,
    ///         Point3::new(-1.0, 0.0, 1.),
    ///         Point3::new(-1.0, -1.0, 1.0) * corner_weight,
    ///         Point3::new(1.0, -1.0, 1.0) * corner_weight,
    ///         Point3::new(1.0, 0.0, 1.),
    ///     ],
    ///     vec![0., 0., 0., 1. / 4., 1. / 2., 1. / 2., 3. / 4., 1., 1., 1.],
    /// ).unwrap();
    /// let approx = unit_circle.try_length().unwrap();
    /// let goal = 2.0 * std::f64::consts::PI; // circumference of the unit circle
    /// assert_relative_eq!(approx, goal);
    /// ```
    pub fn try_length(&self) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let mult = self.knots.multiplicity();
        let start = mult.first().unwrap().multiplicity();
        let end = mult.last().unwrap().multiplicity();

        let segments = self.try_decompose_bezier_segments()?;

        // If the start/end parts of the knot vector are not duplicated,
        // the Bezier segments will not be generated correctly,
        // so reduce the number of segments by the amount that falls below the required duplication degree.
        let required_multiplicity = self.degree + 1;
        let i = if start < required_multiplicity {
            required_multiplicity - start
        } else {
            0
        };
        let j = if end < required_multiplicity {
            segments.len() - (required_multiplicity - end)
        } else {
            segments.len()
        };
        let segments = &segments[i..j];

        let (_, u) = self.knots_domain();
        let gauss = GaussLegendre::init(16 + self.degree);
        let length = segments
            .iter()
            .map(|s| compute_bezier_segment_length(s, u, &gauss))
            .reduce(T::add)
            .unwrap();
        Ok(length)
    }

    /// Divide a NURBS curve by a given length
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    /// let corner_weight = 1. / 2.;
    /// let unit_circle = NurbsCurve2D::try_new(
    ///     2,
    ///     vec![
    ///         Point3::new(1.0, 0.0, 1.),
    ///         Point3::new(1.0, 1.0, 1.0) * corner_weight,
    ///         Point3::new(-1.0, 1.0, 1.0) * corner_weight,
    ///         Point3::new(-1.0, 0.0, 1.),
    ///         Point3::new(-1.0, -1.0, 1.0) * corner_weight,
    ///         Point3::new(1.0, -1.0, 1.0) * corner_weight,
    ///         Point3::new(1.0, 0.0, 1.),
    ///     ],
    ///     vec![0., 0., 0., 1. / 4., 1. / 2., 1. / 2., 3. / 4., 1., 1., 1.],
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        anyhow::ensure!(length > T::zero(), "The length must be greater than zero");

        let segments = self.try_decompose_bezier_segments()?;
        let lengthes = segments
            .iter()
            .map(|s| s.try_length())
            .collect::<anyhow::Result<Vec<_>>>()?;
        let total = lengthes.iter().fold(T::zero(), |a, b| a + *b);

        anyhow::ensure!(
            total > length,
            "The curve is too short to divide by the given length"
        );

        let mut samples = vec![CurveLengthParameter::new(self.knots.first(), T::zero())];

        let mut i = 0;
        let mut lc = length;

        let mut acc = T::zero();
        let mut acc_prev = T::zero();

        let gauss = GaussLegendre::init(16 + self.degree);
        let eps = T::from_f64(1e-6).unwrap();
        let tolerance = T::from_f64(1e-3 * 2.5).unwrap();

        while i < segments.len() {
            let current_length = lengthes[i];
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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
        DefaultAllocator: Allocator<T, D>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<T, <<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let n = points.len();
        if n < degree + 1 {
            anyhow::bail!("Too few control points for curve");
        }

        let pts: Vec<_> = (0..(n + degree))
            .map(|i| {
                if i < n {
                    points[i].clone()
                } else {
                    points[i - n].clone()
                }
            })
            .collect();

        let knots = (0..(n + 1 + degree * 2)).map(|i| T::from_usize(i).unwrap());

        Ok(Self {
            control_points: pts
                .iter()
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
        DefaultAllocator: Allocator<T, D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        Self::try_interpolate_with_tangents(points, degree, None, None)
    }

    /// Try to create an interpolated NURBS curve from a set of points with start and end tangents
    pub fn try_interpolate_with_tangents(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
        start_tangent: Option<OVector<T, DimNameDiff<D, U1>>>,
        end_tangent: Option<OVector<T, DimNameDiff<D, U1>>>,
    ) -> anyhow::Result<Self>
    where
        DefaultAllocator: Allocator<T, D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
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

        let has_tangent = start_tangent.is_some() && end_tangent.is_some();
        let (start, end) = if has_tangent {
            (0, us.len() - degree + 1)
        } else {
            (1, us.len() - degree)
        };

        for i in start..end {
            let mut weight_sums = T::zero();
            for j in 0..degree {
                weight_sums += us[i + j];
            }
            knots_start.push(weight_sums / T::from_usize(degree).unwrap());
        }

        let knots = KnotVector::new([knots_start, vec![T::one(); degree + 1]].concat());
        let plen = points.len();

        let (n, ld) = if has_tangent {
            (plen + 1, plen - (degree - 1))
        } else {
            (plen - 1, plen - (degree + 1))
        };

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
            for j in 0..e.len() {
                m_a[(i, j)] = e[j];
            }
        }

        // dbg!(&mA);

        if has_tangent {
            let cols = m_a.ncols();
            let ln = cols - 2;
            let tan_row0 = [vec![-T::one(), T::one()], vec![T::zero(); ln]].concat();
            let tan_row1 = [vec![T::zero(); ln], vec![-T::one(), T::one()]].concat();
            // dbg!(&tan_row0);
            // dbg!(&tan_row1);
            m_a = m_a.insert_row(1, T::zero());
            let rows = m_a.nrows();
            m_a = m_a.insert_row(rows - 1, T::zero());
            for i in 0..cols {
                m_a[(1, i)] = tan_row0[i];
                m_a[(rows - 1, i)] = tan_row1[i];
            }
        }

        let dim = D::dim() - 1;

        let mult0 = knots[degree + 1] / T::from_usize(degree).unwrap();
        let mult1 = (T::one() - knots[knots.len() - degree - 2]) / T::from_usize(degree).unwrap();

        // solve Ax = b with LU decomposition
        let lu = m_a.lu();
        let mut m_x = DMatrix::<T>::identity(if has_tangent { plen + 2 } else { plen }, dim);
        let rows = m_x.nrows();
        for i in 0..dim {
            let b: Vec<_> = if has_tangent {
                let st = start_tangent.as_ref().unwrap()[i];
                let et = end_tangent.as_ref().unwrap()[i];
                let mut b = vec![points[0].coords[i]];
                b.push(mult0 * st);
                for j in 1..(plen - 1) {
                    b.push(points[j].coords[i]);
                }
                b.push(mult1 * et);
                b.push(points[plen - 1].coords[i]);
                b
            } else {
                points.iter().map(|p| p.coords[i]).collect()
            };

            let b = DVector::from_vec(b);
            // dbg!(&b);
            let xs = lu.solve(&b).ok_or(anyhow::anyhow!("Solve failed"))?;
            for j in 0..rows {
                m_x[(j, i)] = xs[j];
            }
        }

        // dbg!(&mX.shape());

        // extract control points from solved x
        let mut control_points = vec![];
        for i in 0..m_x.nrows() {
            let mut coords = vec![];
            for j in 0..m_x.ncols() {
                coords.push(m_x[(i, j)]);
            }
            coords.push(T::one());
            control_points.push(OPoint::from_slice(&coords));
        }

        // dbg!(control_points.len());
        // dbg!(knots.len());

        Ok(Self {
            degree,
            control_points,
            knots,
        })
    }

    /// Elevate the dimension of the curve (e.g., 2D -> 3D)
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// let points: Vec<Point2<f64>> = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(-1.0, 1.0),
    /// ];
    /// let curve2d = NurbsCurve2D::try_interpolate(&points, 3).unwrap();
    /// let curve3d: NurbsCurve3D<f64> = curve2d.elevate_dimension();
    /// let (start, end) = curve2d.knots_domain();
    /// let (p0, p1) = (curve2d.point_at(start), curve2d.point_at(end));
    /// let (p2, p3) = (curve3d.point_at(start), curve3d.point_at(end));
    /// assert_eq!(p0.x, p2.x);
    /// assert_eq!(p0.y, p2.y);
    /// assert_eq!(p2.z, 0.0);
    /// assert_eq!(p1.x, p3.x);
    /// assert_eq!(p1.y, p3.y);
    /// assert_eq!(p3.z, 0.0);
    /// ```
    pub fn elevate_dimension(&self) -> NurbsCurve<T, DimNameSum<D, U1>>
    where
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<T, DimNameSum<D, U1>>,
    {
        let mut control_points = vec![];
        for p in self.control_points.iter() {
            let mut coords = vec![];
            for i in 0..(D::dim() - 1) {
                coords.push(p[i]);
            }
            coords.push(T::zero()); // set a zero in the last dimension
            coords.push(p[D::dim() - 1]);
            control_points.push(OPoint::from_slice(&coords));
        }

        NurbsCurve {
            control_points,
            degree: self.degree,
            knots: self.knots.clone(),
        }
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
                let numer = ub - ua;
                let mut alfs = vec![T::zero(); new_degree];
                let mut k = new_degree;
                while k > mul {
                    alfs[k - mul - 1] = numer / (knots[a + k] - ua);
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
        let start = if idx > k { idx - k } else { 0 };
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

    /// Check if the curve is clamped
    pub fn is_clamped(&self) -> bool {
        self.knots.is_clamped(self.degree)
    }

    /// Try to refine the curve by inserting knots
    pub fn try_refine_knot(&mut self, knots_to_insert: Vec<T>) -> anyhow::Result<()> {
        anyhow::ensure!(self.is_clamped(), "Curve must be clamped to refine knots");

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
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
        T: ArgminFloat,
        T: ArgminScaledSub<T, T, T>,
    {
        self.find_closest_parameter(point).map(|u| self.point_at(u))
    }

    /// Find the closest parameter on the curve to a given point with Newton's method
    pub fn find_closest_parameter(&self, point: &OPoint<T, DimNameDiff<D, U1>>) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
        T: ArgminFloat,
        T: ArgminScaledSub<T, T, T>,
    {
        let (min_u, max_u) = self.knots_domain();
        let samples = self.control_points.len() * self.degree;
        let pts = self.sample_regular_range_with_parameter(min_u, max_u, samples);

        let mut min = <T as RealField>::max_value().unwrap();
        let mut u = min_u;

        let closed =
            (&self.control_points[0] - &self.control_points[self.control_points.len() - 1]).norm()
                < T::default_epsilon();

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

        let solver = ClosestParameterNewton::new((min_u, max_u), closed);
        let res = Executor::new(ClosestParameterProblem::new(point, self), solver)
            .configure(|state| state.param(u).max_iters(5))
            .run()?;
        res.state()
            .get_best_param()
            .cloned()
            .ok_or(anyhow::anyhow!("No best parameter found"))

        /*
        // old newton method implementation
        let mut cu = u;

        let max_iterations = 5;
        for _ in 0..max_iterations {
            let e = self.rational_derivatives(cu, 2);
            let dif = &e[0] - &point.coords;

            let c1v = dif.norm();

            let c2n = e[1].dot(&dif);
            let c2d = e[1].norm() * c1v;

            let c2v = c2n / c2d;

            let c1 = c1v < T::default_epsilon();
            let c2 = c2v.abs() < T::default_epsilon();

            if c1 && c2 {
                return cu;
            }

            let f = c2n;
            let s0 = e[2].dot(&dif);
            let s1 = e[1].dot(&e[1]);
            let df = s0 + s1;
            let mut ct = cu - f / df;

            if ct < min_u {
                ct = if closed { max_u - (ct - min_u) } else { min_u };
            } else if ct > max_u {
                ct = if closed { min_u + (ct - max_u) } else { max_u };
            }

            let c3v = (&e[1] * (ct - cu)).norm();
            if c3v < T::default_epsilon() {
                return cu;
            }

            cu = ct;
        }
        cu
        */
    }

    pub fn find_intersections(
        &self,
        other: &Self,
    ) -> anyhow::Result<Vec<CurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let _eps = T::from_f64(1e-8).unwrap();
        let traversed = BoundingBoxTraversal::try_traverse(self, other, None)?;

        let pts = traversed
            .into_pairs_iter()
            .filter_map(|(a, b)| {
                let _ca = a.curve_owned();
                let _cb = b.curve_owned();

                // gauss-newton line search example
                // https://github.com/argmin-rs/argmin/blob/main/examples/gaussnewton_linesearch/src/main.rs

                /*
                let problem = CurveIntersectionProblem::new(&ca, &cb);
                let init_param = OVector::<T, U2>::new(ca.knots_domain().0, cb.knots_domain().0);
                let solver = CurveIntersectionNewton::<T>::new();
                let res = Executor::new(problem, solver)
                    .configure(|state| state.param(init_param).max_iters(5))
                    .run();
                */

                /*
                res.state()
                    .get_best_param()
                    .cloned()
                    .ok_or(anyhow::anyhow!("No best parameter found"));
                */

                todo!()

                /*
                let init_cost = problem.cost(&init_param).unwrap();
                let init_grad = problem.gradient(&init_param).unwrap();
                solver.search_direction(-init_grad);
                solver.initial_step_length(T::one());
                // solver.initial_step_length(T::from_f64(5.).unwrap());
                let res = Executor::new(problem, solver)
                    .configure(|state| {
                        state
                            .param(init_param)
                            // .cost(init_cost)
                            // .gradient(init_grad)
                            .max_iters(1000)
                    })
                    .run();
                // dbg!(init_cost, init_grad);
                if let Err(err) = res {
                    dbg!(err);
                    return None;
                }
                res.ok().and_then(|r| {
                    dbg!(init_param, r.state().get_best_param());
                    r.state().get_best_param().map(|param| {
                        let p0 = ca.point_at(param[0]);
                        let p1 = cb.point_at(param[1]);
                        CurveIntersection::new((p0, param[0]), (p1, param[1]))
                    })
                })
                */
            })
            .filter(|_it| {
                true
                /*
                let p0 = &it.a().0;
                let p1 = &it.b().0;
                let d = (p0 - p1).norm_squared();
                d < eps
                */
            })
            .collect();

        Ok(pts)
    }

    /// Trim the curve into two curves before and after the parameter
    pub fn try_trim(&self, u: T) -> anyhow::Result<(Self, Self)> {
        let knots_to_insert: Vec<_> = (0..=self.degree).map(|_| u).collect();
        let mut cloned = self.clone();
        cloned.try_refine_knot(knots_to_insert)?;

        let n = self.knots.len() - self.degree - 2;
        let s = self.knots.find_knot_span_index(n, self.degree, u);
        let knots0 = cloned.knots.as_slice()[0..=(s + self.degree + 1)].to_vec();
        let knots1 = cloned.knots.as_slice()[s + 1..].to_vec();
        let cpts0 = cloned.control_points[0..=s].to_vec();
        let cpts1 = cloned.control_points[s + 1..].to_vec();
        Ok((
            Self {
                degree: self.degree,
                control_points: cpts0,
                knots: KnotVector::new(knots0),
            },
            Self {
                degree: self.degree,
                control_points: cpts1,
                knots: KnotVector::new(knots1),
            },
        ))
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
        /*
        anyhow::ensure!(
            self.is_clamped(),
            "Curve must be clamped to decompose into Bezier segments"
        );
        */

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

        Ok(segments)
    }

    /// Cast the curve to a curve with another floating point type
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> NurbsCurve<F, D>
    where
        DefaultAllocator: Allocator<F, D>,
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
}

/// Enable to transform a NURBS curve by a given DxD matrix
impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for NurbsCurve<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.control_points.iter_mut().for_each(|p| {
            let mut pt = *p;
            pt[D - 1] = T::one();
            let transformed = transform * pt;
            let w = transformed[D - 1];
            for i in 0..D - 1 {
                p[i] = transformed[i] / w;
            }
        });
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<T, D>,
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
fn compute_bezier_segment_parameter_at_length<T: FloatingPoint, D: DimName>(
    s: &NurbsCurve<T, D>,
    length: T,
    tolerance: T,
    total_length: T,
    gauss: &GaussLegendre,
) -> T
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    let (k0, k1) = s.knots_domain();
    if length < T::zero() {
        return k0;
    } else if length > total_length {
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
fn compute_bezier_segment_length<T: FloatingPoint, D: DimName>(
    s: &NurbsCurve<T, D>,
    u: T,
    gauss: &GaussLegendre,
) -> T
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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
pub fn dehomogenize<T: FloatingPoint, D: DimName>(
    point: &OPoint<T, D>,
) -> Option<OPoint<T, DimNameDiff<D, U1>>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
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

#[cfg(test)]
mod tests {}
