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
    derive(serde::Serialize),
    serde(bound(serialize = "T: serde::Serialize, NurbsCurve<T, D>: serde::Serialize"))
)]
pub struct TrimmedCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    curve: NurbsCurve<T, D>,
    /// Active parameter domain. If None, uses the full knot domain.
    domain: Option<(T, T)>,
}

/// Also accepts the legacy form where a span was a bare `NurbsCurve` (read as a
/// full-domain trimmed curve), so older serialized data still loads.
#[cfg(feature = "serde")]
impl<'de, T, D> serde::Deserialize<'de> for TrimmedCurve<T, D>
where
    T: FloatingPoint + serde::Deserialize<'de>,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    NurbsCurve<T, D>: serde::Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(
            bound(
                deserialize = "T: serde::Deserialize<'de>, NurbsCurve<T, D>: serde::Deserialize<'de>"
            ),
            untagged
        )]
        enum Repr<T: FloatingPoint, D: DimName>
        where
            DefaultAllocator: Allocator<D>,
        {
            Trimmed {
                curve: NurbsCurve<T, D>,
                #[serde(default)]
                domain: Option<(T, T)>,
            },
            Bare(NurbsCurve<T, D>),
        }

        Ok(match Repr::<T, D>::deserialize(deserializer)? {
            Repr::Trimmed { curve, domain } => Self { curve, domain },
            Repr::Bare(curve) => Self {
                curve,
                domain: None,
            },
        })
    }
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

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::{TrimmedCurve, TrimmedCurve3D};
    use crate::prelude::NurbsCurve3D;
    use nalgebra::Point3;

    #[test]
    fn deserialize_accepts_legacy_and_current_forms() {
        let curve = NurbsCurve3D::polyline(&[Point3::origin(), Point3::new(1.0, 0.0, 0.0)], true);

        // Legacy form: a span serialized as a bare NurbsCurve loads as full-domain.
        let legacy_json = serde_json::to_string(&curve).unwrap();
        let from_legacy: TrimmedCurve3D<f64> = serde_json::from_str(&legacy_json).unwrap();
        assert!(from_legacy.is_full_domain());
        assert_eq!(from_legacy.curve(), &curve);

        // Current form: { curve, domain } round-trips.
        let wrapped = TrimmedCurve::from_curve(curve);
        let wrapped_json = serde_json::to_string(&wrapped).unwrap();
        let back: TrimmedCurve3D<f64> = serde_json::from_str(&wrapped_json).unwrap();
        assert_eq!(back, wrapped);
    }
}
