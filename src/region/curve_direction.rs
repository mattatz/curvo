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
    pub fn new<T: FloatingPoint, D: DimName>(
        a: &NurbsCurve<T, D>,
        b: &NurbsCurve<T, D>,
        epsilon: T,
    ) -> Option<Self>
    where
        DefaultAllocator: Allocator<D>,
        D: DimNameSub<U1>,
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

        directions
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .and_then(|min| if min.0 < epsilon { Some(min.1) } else { None })
    }
}
