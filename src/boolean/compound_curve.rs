use argmin::core::ArgminFloat;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator};

use crate::{
    curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions,
    region::CompoundCurve,
};

use super::clip::{clip, Clipped};
use super::Boolean;

/// Boolean operation for compound curve & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for CompoundCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    // type Output = anyhow::Result<Vec<Region<T>>>;
    type Output = anyhow::Result<Clipped<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: super::operation::BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(other, option.clone())?;
        clip(self, other, operation, option, intersections)
    }
}
