use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

use super::{has_intersection_parameter::HasIntersectionParameter, CurveIntersection};

type Intersection<'a, T, D> = (&'a NurbsCurve<T, D>, OPoint<T, DimNameDiff<D, U1>>, T);

/// A struct representing the intersection of two curves.
#[derive(Debug)]
pub struct CompoundCurveIntersection<'a, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    a: Intersection<'a, T, D>,
    b: Intersection<'a, T, D>,
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

    pub fn a(&self) -> &Intersection<'a, T, D> {
        &self.a
    }

    pub fn b(&self) -> &Intersection<'a, T, D> {
        &self.b
    }
}

impl<'a, T: FloatingPoint, D: DimName> HasIntersectionParameter<T>
    for CompoundCurveIntersection<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn a_parameter(&self) -> T {
        self.a.2
    }

    fn b_parameter(&self) -> T {
        self.b.2
    }
}
