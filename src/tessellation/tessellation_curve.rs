use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, RealField, U1,
};
use rand::{rngs::ThreadRng, RngExt};

use crate::{
    curve::NurbsCurve,
    misc::{three_points_are_flat, FloatingPoint},
};

use super::Tessellation;

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

        let mut rng = rand::rng();
        let tol = tolerance.unwrap_or(T::from_f64(1e-6).unwrap());
        let (start, end) = self.knots_domain();
        tessellate_curve_adaptive(self, start, end, tol, &mut rng, &|_t, p| p)
    }
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

/// Tessellate the curve using an adaptive algorithm recursively
/// if the curve between [start ~ end] is flat enough, it will return the two end points
/// f is a function that maps the t and point to a new type P
pub(crate) fn tessellate_curve_adaptive<T: FloatingPoint, D, P, F>(
    curve: &NurbsCurve<T, D>,
    start: T,
    end: T,
    tol: T,
    rng: &mut ThreadRng,
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

    let t = 0.5_f64 + 0.2_f64 * rng.random::<f64>();
    let mid = start + delta * T::from_f64(t).unwrap();
    let p2 = curve.point_at(mid);

    let diff = &p1 - &p3;
    let diff2 = &p1 - &p2;
    if (diff.dot(&diff) < tol && diff2.dot(&diff2) > tol)
        || !three_points_are_flat(&p1, &p2, &p3, tol)
    {
        let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
        let mut left_pts = tessellate_curve_adaptive(curve, start, exact_mid, tol, rng, f);
        let right_pts = tessellate_curve_adaptive(curve, exact_mid, end, tol, rng, f);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![f(start, p1), f(end, p3)]
    }
}

/// Tessellate the curve using a length-based adaptive algorithm.
///
/// Subdivides each segment until its chord length is at most
/// `max_edge_length`, expressed in the curve's world units. The midpoint
/// parameter is fixed at 0.5 to keep the tessellation deterministic.
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
    let chord = &p3 - &p1;
    let chord_len = chord.norm();

    if chord_len > max_edge_length {
        let mid = start + delta * T::from_f64(0.5).unwrap();
        let mut left_pts = tessellate_curve_adaptive_length(curve, start, mid, max_edge_length, f);
        let right_pts = tessellate_curve_adaptive_length(curve, mid, end, max_edge_length, f);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![f(start, p1), f(end, p3)]
    }
}
