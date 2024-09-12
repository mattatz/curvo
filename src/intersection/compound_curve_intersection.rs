use std::borrow::{Borrow, Cow};

use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

use super::{has_intersection::HasIntersection, CurveIntersection, HasIntersectionParameter};

type Intersection<'a, T, D> = (Cow<'a, NurbsCurve<T, D>>, OPoint<T, DimNameDiff<D, U1>>, T);

/// A struct representing the intersection of two curves.
#[derive(Debug, Clone)]
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
            a: (Cow::Borrowed(a), pa, ta),
            b: (Cow::Borrowed(b), pb, tb),
        }
    }

    pub fn a_curve(&self) -> &NurbsCurve<T, D> {
        self.a.0.borrow()
    }

    pub fn b_curve(&self) -> &NurbsCurve<T, D> {
        self.b.0.borrow()
    }

    pub fn swap(&mut self) {
        std::mem::swap(&mut self.a, &mut self.b);
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

impl<'a, T: FloatingPoint, D: DimName> HasIntersection<Intersection<'a, T, D>, T>
    for CompoundCurveIntersection<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn a(&self) -> &Intersection<'a, T, D> {
        &self.a
    }

    fn b(&self) -> &Intersection<'a, T, D> {
        &self.b
    }
}
