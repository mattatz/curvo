use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, RealField, U1,
};

use crate::{
    curve::NurbsCurve,
    misc::{three_points_are_flat, FloatingPoint},
};

use super::{ParametricTessellation, Tessellation};

impl<T: FloatingPoint, D: DimName> Tessellation<Option<T>> for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<OPoint<T, DimNameDiff<D, U1>>>;
    /// Tessellate the curve using an adaptive algorithm
    /// this `adaptive` means that the curve will be tessellated based on the curvature of the curve
    fn tessellate(&self, tolerance: Option<T>) -> Self::Output {
        if self.degree() == 1 {
            return self.dehomogenized_control_points();
        }

        let tol = tolerance.unwrap_or(T::from_f64(1e-6).unwrap());
        let (start, end) = self.knots_domain();
        let mut sampler = FlatnessSampler::default();
        tessellate_curve_adaptive(self, start, end, tol, &mut sampler, &|_t, p| p)
    }
}

impl<T: FloatingPoint, D: DimName> ParametricTessellation<Option<T>> for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<(T, OPoint<T, DimNameDiff<D, U1>>)>;
    /// Tessellate the curve using an adaptive algorithm,
    /// returning the parameter and the point of each tessellated vertex.
    /// The points are identical to the ones returned by [`Tessellation::tessellate`].
    fn tessellate_with_parameters(&self, tolerance: Option<T>) -> Self::Output {
        if self.degree() == 1 {
            return polyline_with_parameters(self);
        }

        let tol = tolerance.unwrap_or(T::from_f64(1e-6).unwrap());
        let (start, end) = self.knots_domain();
        let mut sampler = FlatnessSampler::default();
        tessellate_curve_adaptive(self, start, end, tol, &mut sampler, &|t, p| (t, p))
    }
}

/// Returns the control points of a degree 1 curve paired with their parameters.
/// The control point `i` of a degree 1 curve is located at the knot `i + 1`.
#[allow(clippy::type_complexity)]
fn polyline_with_parameters<T: FloatingPoint, D>(
    curve: &NurbsCurve<T, D>,
) -> Vec<(T, OPoint<T, DimNameDiff<D, U1>>)>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let knots = curve.knots().as_slice();
    curve
        .dehomogenized_control_points()
        .into_iter()
        .enumerate()
        .map(|(i, p)| (knots[i + 1], p))
        .collect()
}

/// Options for length-based adaptive curve tessellation.
///
/// Subdivides each segment until its chord length is at most
/// `max_edge_length`. The value is interpreted in the curve's world units
/// (e.g. mm).
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveCurveTessellationOptions<T> {
    /// Maximum allowed segment chord length, in the curve's world units.
    pub max_edge_length: T,
}

impl<T: RealField> Default for AdaptiveCurveTessellationOptions<T> {
    fn default() -> Self {
        Self {
            max_edge_length: T::from_f64(1.0).unwrap(),
        }
    }
}

impl<T: RealField> AdaptiveCurveTessellationOptions<T> {
    pub fn new(max_edge_length: T) -> Self {
        Self { max_edge_length }
    }

    pub fn with_max_edge_length(mut self, max_edge_length: T) -> Self {
        self.max_edge_length = max_edge_length;
        self
    }
}

impl<T: FloatingPoint, D: DimName> Tessellation<AdaptiveCurveTessellationOptions<T>>
    for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<OPoint<T, DimNameDiff<D, U1>>>;

    /// Tessellate the curve using a length-based adaptive algorithm.
    ///
    /// Subdivides each segment until its chord length does not exceed
    /// `max_edge_length`, expressed in the curve's world units.
    fn tessellate(&self, options: AdaptiveCurveTessellationOptions<T>) -> Self::Output {
        if self.degree() == 1 {
            return self.dehomogenized_control_points();
        }

        let (start, end) = self.knots_domain();
        tessellate_curve_adaptive_length(self, start, end, options.max_edge_length, &|_t, p| p)
    }
}

impl<T: FloatingPoint, D: DimName> ParametricTessellation<AdaptiveCurveTessellationOptions<T>>
    for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<(T, OPoint<T, DimNameDiff<D, U1>>)>;

    /// Tessellate the curve using a length-based adaptive algorithm,
    /// returning the parameter and the point of each tessellated vertex.
    /// The points are identical to the ones returned by [`Tessellation::tessellate`].
    fn tessellate_with_parameters(
        &self,
        options: AdaptiveCurveTessellationOptions<T>,
    ) -> Self::Output {
        if self.degree() == 1 {
            return polyline_with_parameters(self);
        }

        let (start, end) = self.knots_domain();
        tessellate_curve_adaptive_length(self, start, end, options.max_edge_length, &|t, p| (t, p))
    }
}

/// Deterministic sequence of the sampling parameter used for the flatness test
/// in [`tessellate_curve_adaptive`].
///
/// Sampling off the exact midpoint prevents a curved span from looking flat
/// when the midpoint sits on a symmetry of the span.
/// The parameter is generated by the golden ratio additive recurrence (a low-discrepancy sequence)
/// instead of a random number generator, so that the tessellation is reproducible.
#[derive(Clone, Debug, Default)]
pub(crate) struct FlatnessSampler {
    state: f64,
}

impl FlatnessSampler {
    /// Returns the next sampling parameter in [0.5, 0.7)
    fn next(&mut self) -> f64 {
        const GOLDEN_RATIO_CONJUGATE: f64 = 0.618_033_988_749_894_9;
        self.state = (self.state + GOLDEN_RATIO_CONJUGATE).fract();
        0.5 + 0.2 * self.state
    }
}

/// Tessellate the curve using an adaptive algorithm recursively
/// if the curve between [start ~ end] is flat enough, it will return the two end points
/// f is a function that maps the t and point to a new type P
pub(crate) fn tessellate_curve_adaptive<T: FloatingPoint, D, P, F>(
    curve: &NurbsCurve<T, D>,
    start: T,
    end: T,
    tol: T,
    sampler: &mut FlatnessSampler,
    f: &F,
) -> Vec<P>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    F: Fn(T, OPoint<T, DimNameDiff<D, U1>>) -> P,
    P: Clone,
{
    let p1 = curve.point_at(start);
    let delta = end - start;
    if delta < T::from_f64(1e-8).unwrap() {
        return vec![f(start, p1)];
    }

    let p3 = curve.point_at(end);

    let t = sampler.next();
    let mid = start + delta * T::from_f64(t).unwrap();
    let p2 = curve.point_at(mid);

    let diff = &p1 - &p3;
    let diff2 = &p1 - &p2;
    if (diff.dot(&diff) < tol && diff2.dot(&diff2) > tol)
        || !three_points_are_flat(&p1, &p2, &p3, tol)
    {
        let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
        let mut left_pts = tessellate_curve_adaptive(curve, start, exact_mid, tol, sampler, f);
        let right_pts = tessellate_curve_adaptive(curve, exact_mid, end, tol, sampler, f);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![f(start, p1), f(end, p3)]
    }
}

/// Tessellate the curve using a length-based adaptive algorithm.
///
/// Subdivides each segment until its chord length is at most
/// `max_edge_length`, expressed in the curve's world units. A span is subdivided
/// if the chord or either half-chord to its midpoint exceeds `max_edge_length`.
/// The midpoint parameter is fixed at 0.5 to keep the tessellation deterministic.
pub(crate) fn tessellate_curve_adaptive_length<T: FloatingPoint, D, P, F>(
    curve: &NurbsCurve<T, D>,
    start: T,
    end: T,
    max_edge_length: T,
    f: &F,
) -> Vec<P>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    F: Fn(T, OPoint<T, DimNameDiff<D, U1>>) -> P,
    P: Clone,
{
    let p1 = curve.point_at(start);
    let delta = end - start;
    if delta < T::from_f64(1e-8).unwrap() {
        return vec![f(start, p1)];
    }

    let p3 = curve.point_at(end);
    let mid = start + delta * T::from_f64(0.5).unwrap();
    let p2 = curve.point_at(mid);

    // Also check the chords to the midpoint, so that a span whose end points coincide
    // (e.g. a closed curve) is still subdivided.
    let exceeds = (&p3 - &p1).norm() > max_edge_length
        || (&p2 - &p1).norm() > max_edge_length
        || (&p3 - &p2).norm() > max_edge_length;

    if exceeds {
        let mut left_pts = tessellate_curve_adaptive_length(curve, start, mid, max_edge_length, f);
        let right_pts = tessellate_curve_adaptive_length(curve, mid, end, max_edge_length, f);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![f(start, p1), f(end, p3)]
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Point4};

    use crate::prelude::*;

    fn bezier() -> NurbsCurve3D<f64> {
        NurbsCurve3D::try_new(
            3,
            vec![
                Point4::new(0.0, 0.0, 0.0, 1.0),
                Point4::new(1.0, 2.0, 0.0, 1.0),
                Point4::new(2.0, -2.0, 0.0, 1.0),
                Point4::new(3.0, 0.0, 0.0, 1.0),
            ],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap()
    }

    fn closed_curve() -> NurbsCurve3D<f64> {
        let points = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
        ];
        NurbsCurve3D::try_periodic(&points, 3).unwrap()
    }

    #[test]
    fn tessellate_is_deterministic() {
        for curve in [bezier(), closed_curve()] {
            let first = curve.tessellate(Some(1e-6));
            for _ in 0..8 {
                assert_eq!(curve.tessellate(Some(1e-6)), first);
            }
        }
    }

    fn assert_parameters<O: Clone>(curve: &NurbsCurve3D<f64>, options: O)
    where
        NurbsCurve3D<f64>: Tessellation<O, Output = Vec<Point3<f64>>>
            + ParametricTessellation<O, Output = Vec<(f64, Point3<f64>)>>,
    {
        let points = curve.tessellate(options.clone());
        let with_parameters = curve.tessellate_with_parameters(options);
        assert_eq!(points.len(), with_parameters.len());

        let (start, end) = curve.knots_domain();
        assert_eq!(with_parameters.first().unwrap().0, start);
        assert_eq!(with_parameters.last().unwrap().0, end);
        for w in with_parameters.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        for (p, (t, q)) in points.iter().zip(with_parameters.iter()) {
            assert_eq!(p, q);
            assert_relative_eq!(curve.point_at(*t), *q, epsilon = 1e-12);
        }
    }

    #[test]
    fn tessellate_with_parameters() {
        let polyline = NurbsCurve3D::polyline(
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
            ],
            true,
        );
        for curve in [bezier(), closed_curve(), polyline] {
            assert_parameters(&curve, Some(1e-6));
            assert_parameters(&curve, AdaptiveCurveTessellationOptions::new(0.05));
        }
    }

    #[test]
    fn tessellate_closed_curve_by_length() {
        let curve = closed_curve();
        let max_edge_length = 0.05;
        let pts = curve.tessellate(AdaptiveCurveTessellationOptions::new(max_edge_length));
        let length = curve.try_length().unwrap();
        assert!(pts.len() as f64 > length / max_edge_length);
        for w in pts.windows(2) {
            assert!((w[1] - w[0]).norm() <= max_edge_length);
        }
    }
}
