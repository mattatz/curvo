use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, U1,
};

use crate::misc::FloatingPoint;

use super::NurbsCurve;

/// A NURBS curve restricted to a parameter sub-interval.
///
/// Domain is an interval `[start, end]` that may be a subset of the curve's knot domain. If None, uses the full knot domain.
/// The underlying curve is never split, preserving exact geometry.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound(
        serialize = "T: serde::Serialize, NurbsCurve<T, D>: serde::Serialize",
        deserialize = "T: serde::Deserialize<'de>, NurbsCurve<T, D>: serde::Deserialize<'de>"
    ))
)]
pub struct TrimmedCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    curve: NurbsCurve<T, D>,
    /// Active parameter domain. If None, uses the full knot domain.
    domain: Option<(T, T)>,
}

impl<T: FloatingPoint, D: DimName> TrimmedCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a trimmed curve with a specific parameter domain.
    pub fn new(curve: NurbsCurve<T, D>, domain: (T, T)) -> Self {
        Self {
            curve,
            domain: Some(domain),
        }
    }

    /// Create a trimmed curve that uses the full knot domain (no trimming).
    pub fn from_curve(curve: NurbsCurve<T, D>) -> Self {
        Self {
            curve,
            domain: None,
        }
    }

    /// Get the underlying NURBS curve.
    pub fn curve(&self) -> &NurbsCurve<T, D> {
        &self.curve
    }

    /// Get the underlying NURBS curve mutably.
    pub fn curve_mut(&mut self) -> &mut NurbsCurve<T, D> {
        &mut self.curve
    }

    /// Consume self and return the underlying curve.
    pub fn into_curve(self) -> NurbsCurve<T, D> {
        self.curve
    }

    /// Get the active parameter domain.
    /// Returns the explicit domain if set, otherwise the curve's knot domain.
    pub fn domain(&self) -> (T, T) {
        self.domain.unwrap_or_else(|| self.curve.knots_domain())
    }

    /// Get the explicit domain override, if any.
    pub fn explicit_domain(&self) -> Option<(T, T)> {
        self.domain
    }

    /// Set the active parameter domain.
    pub fn set_domain(&mut self, domain: Option<(T, T)>) {
        self.domain = domain;
    }

    /// Check if the domain covers the full curve knot domain.
    pub fn is_full_domain(&self) -> bool {
        self.domain.is_none()
    }

    /// Get the degree of the underlying curve.
    pub fn degree(&self) -> usize {
        self.curve.degree()
    }

    /// Alias for `domain()` for compatibility with NurbsCurve API.
    pub fn knots_domain(&self) -> (T, T) {
        self.domain()
    }

    /// Get the knot vector of the underlying curve.
    pub fn knots(&self) -> &crate::knot::KnotVector<T> {
        self.curve.knots()
    }

    /// Get the control points of the underlying curve.
    pub fn control_points(&self) -> &Vec<nalgebra::OPoint<T, D>> {
        self.curve.control_points()
    }

    /// Get mutable iterator over control points.
    pub fn control_points_iter_mut(&mut self) -> impl Iterator<Item = &mut nalgebra::OPoint<T, D>> {
        self.curve.control_points_iter_mut()
    }

    /// Get the weights of the underlying curve.
    pub fn weights(&self) -> Vec<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        self.curve.weights()
    }

    /// Get dehomogenized control points.
    pub fn dehomogenized_control_points(&self) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        self.curve.dehomogenized_control_points()
    }
}

impl<T: FloatingPoint, D: DimName> From<NurbsCurve<T, D>> for TrimmedCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from(curve: NurbsCurve<T, D>) -> Self {
        Self::from_curve(curve)
    }
}

impl<T: FloatingPoint, D: DimName> std::ops::Deref for TrimmedCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Target = NurbsCurve<T, D>;
    fn deref(&self) -> &NurbsCurve<T, D> {
        &self.curve
    }
}

impl<T: FloatingPoint, D: DimName> std::ops::DerefMut for TrimmedCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn deref_mut(&mut self) -> &mut NurbsCurve<T, D> {
        &mut self.curve
    }
}

impl<T: FloatingPoint, D: DimName> TrimmedCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D> + Allocator<DimNameDiff<D, U1>>,
{
    /// Evaluate a point on the curve at parameter `t`.
    /// `t` should be within `self.domain()`.
    pub fn point_at(&self, t: T) -> OPoint<T, DimNameDiff<D, U1>> {
        self.curve.point_at(t)
    }

    /// Get the start point of the trimmed domain.
    pub fn start_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.curve.point_at(self.domain().0)
    }

    /// Get the end point of the trimmed domain.
    pub fn end_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.curve.point_at(self.domain().1)
    }

    /// Get the tangent at parameter `t`.
    pub fn tangent_at(&self, t: T) -> OVector<T, DimNameDiff<D, U1>> {
        self.curve.tangent_at(t)
    }
}

/// Type alias for 2D trimmed NURBS curve (homogeneous: u*w, v*w, w)
pub type TrimmedCurve2D<T> = TrimmedCurve<T, nalgebra::U3>;

/// Type alias for 3D trimmed NURBS curve (homogeneous: x*w, y*w, z*w, w)
pub type TrimmedCurve3D<T> = TrimmedCurve<T, nalgebra::U4>;
