use std::cmp::Ordering;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

/// Direction of the two connected curves.
#[derive(Clone, Copy, Debug)]
pub enum CurveDirection {
    Forward,  // -> ->
    Backward, // <- <-
    Facing,   // -> <-
    Opposite, // <- ->
}

impl CurveDirection {
    pub fn new<T: FloatingPoint, D>(
        a: &NurbsCurve<T, D>,
        b: &NurbsCurve<T, D>,
        epsilon: T,
    ) -> Option<Self>
    where
        D: DimName + DimNameSub<U1>,
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let ad = a.knots_domain();
        let bd = b.knots_domain();
        let (a0, a1) = (a.point_at(ad.0), a.point_at(ad.1));
        let (b0, b1) = (b.point_at(bd.0), b.point_at(bd.1));
        let directions = [
            ((&a1 - &b0).norm(), Self::Forward),
            ((&a0 - &b1).norm(), Self::Backward),
            ((&a1 - &b1).norm(), Self::Facing),
            ((&a0 - &b0).norm(), Self::Opposite),
        ];

        // Closest direction within epsilon; rounding-level ties fall back to the
        // array's priority order so ~1e-16 noise can't pick an inverting join for a
        // closed loop (start ≈ end) and corrupt the assembled order.
        let tie = epsilon * T::from_f64(1e-3).unwrap();
        directions
            .iter()
            .filter(|(gap, _)| *gap < epsilon)
            .min_by(|a, b| {
                if (a.0 - b.0).abs() <= tie {
                    Ordering::Equal
                } else {
                    a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
                }
            })
            .map(|(_, direction)| *direction)
    }
}

#[cfg(test)]
mod tests {
    use super::CurveDirection;
    use crate::prelude::NurbsCurve3D;
    use nalgebra::Point3;

    const EPS: f64 = 1e-4;

    fn line(a: Point3<f64>, b: Point3<f64>) -> NurbsCurve3D<f64> {
        NurbsCurve3D::polyline(&[a, b], true)
    }

    /// When several directions tie within epsilon (e.g. a curve whose start == end),
    /// the non-inverting `Forward` must win so the assembled chain is not reordered.
    #[test]
    fn tie_prefers_forward() {
        let p = Point3::new(0.0, 0.0, 0.0);
        // `a` is a closed loop (triangle): start == end == p, but with real extent.
        let a = NurbsCurve3D::polyline(
            &[p, Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0), p],
            true,
        );
        // `b` starts at p, so both Forward (a1->b0) and Opposite (a0->b0) gaps are ~0.
        let b = line(p, Point3::new(2.0, 2.0, 0.0));
        assert!(matches!(
            CurveDirection::new(&a, &b, EPS),
            Some(CurveDirection::Forward)
        ));
    }

    /// A genuinely closer join within epsilon must beat a near-miss that is also
    /// within epsilon — the gap magnitude is not discarded.
    #[test]
    fn closest_within_epsilon_wins() {
        // Forward gap = 5e-5 (within epsilon but a near-miss).
        // Backward gap = 0 (exact). Backward must be chosen.
        let a = line(Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0));
        let b = line(Point3::new(5e-5, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        assert!(matches!(
            CurveDirection::new(&a, &b, EPS),
            Some(CurveDirection::Backward)
        ));
    }

    /// Endpoints farther apart than epsilon do not connect.
    #[test]
    fn beyond_epsilon_is_none() {
        let a = line(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0));
        let b = line(Point3::new(2.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
        assert!(CurveDirection::new(&a, &b, EPS).is_none());
    }
}
