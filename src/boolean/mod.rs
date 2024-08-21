use nalgebra::{allocator::Allocator, Const, DefaultAllocator};

use crate::{curve::NurbsCurve, misc::FloatingPoint, region::Region};

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;

    fn union(&self, other: &T) -> Self::Output;
    fn intersection(&self, other: &T) -> Self::Output;
    fn difference(&self, other: &T) -> Self::Output;
}

impl<T: FloatingPoint> Boolean<NurbsCurve<T, Const<3>>> for NurbsCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    type Output = Region<T>;

    fn union(&self, other: &NurbsCurve<T, Const<3>>) -> Self::Output {
        unimplemented!()
    }

    fn intersection(&self, other: &NurbsCurve<T, Const<3>>) -> Self::Output {
        unimplemented!()
    }

    fn difference(&self, other: &NurbsCurve<T, Const<3>>) -> Self::Output {
        unimplemented!()
    }
}
