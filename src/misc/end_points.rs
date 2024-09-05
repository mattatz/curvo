use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{curve::NurbsCurve, region::CompoundCurve};

use super::FloatingPoint;

pub trait EndPoints<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    fn first_point(&self) -> OPoint<T, D>;
    fn end_point(&self) -> OPoint<T, D>;
    fn end_points(&self) -> (OPoint<T, D>, OPoint<T, D>) {
        (self.first_point(), self.end_point())
    }
}

impl<T: FloatingPoint, D: DimName> EndPoints<T, DimNameDiff<D, U1>> for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn first_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.point_at(self.knots_domain().0)
    }

    fn end_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.point_at(self.knots_domain().1)
    }
}

impl<T: FloatingPoint, D: DimName> EndPoints<T, DimNameDiff<D, U1>> for CompoundCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn first_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.spans().first().unwrap().first_point()
    }

    fn end_point(&self) -> OPoint<T, DimNameDiff<D, U1>> {
        self.spans().last().unwrap().end_point()
    }
}
