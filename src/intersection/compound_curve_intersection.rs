use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

use super::CurveIntersection;

/// A struct representing the intersection of two curves.
#[derive(Debug)]
pub struct CompoundCurveIntersection<'a, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    a: (&'a NurbsCurve<T, D>, OPoint<T, DimNameDiff<D, U1>>, T),
    b: (&'a NurbsCurve<T, D>, OPoint<T, DimNameDiff<D, U1>>, T),
}

impl<'a, T: FloatingPoint, D: DimName> CompoundCurveIntersection<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new(
        a: &'a NurbsCurve<T, D>,
        b: &'a NurbsCurve<T, D>,
        intersection: CurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>,
    ) -> Self {
        let ((pa, ta), (pb, tb)) = intersection.as_tuple();
        Self {
            a: (a, pa, ta),
            b: (b, pb, tb),
        }
    }
}
