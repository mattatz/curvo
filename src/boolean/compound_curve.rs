use argmin::core::ArgminFloat;
use nalgebra::U3;

use crate::prelude::Intersection;
use crate::{
    curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions,
    region::CompoundCurve,
};

use super::clip::{clip, Clip};
use super::operation::BooleanOperation;
use super::Boolean;

/// Boolean operation for compound curve & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, U3>> for CompoundCurve<T, U3> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, U3>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(other, option.clone())?;
        clip(self, other, operation, option, intersections)
    }
}

/// Boolean operation for compound curve & compound curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a CompoundCurve<T, U3>>
    for CompoundCurve<T, U3>
{
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a CompoundCurve<T, U3>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(other, option.clone())?;
        clip(self, other, operation, option, intersections)
    }
}
