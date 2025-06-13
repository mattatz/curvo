use std::cmp::Ordering;

use super::clip::{clip, Clip};
use super::operation::BooleanOperation;
use super::Boolean;
use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::U3;

use crate::prelude::{CompoundCurveIntersection, HasIntersectionParameter, Intersects};
use crate::region::CompoundCurve;
use crate::{curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions};

/// Boolean operation for two NURBS curves.
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, U3>> for NurbsCurve<T, U3> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, U3>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersection(other, option.clone())?;
        let intersections = intersections
            .into_iter()
            .map(|it| CompoundCurveIntersection::new(self, other, it))
            .collect_vec();
        clip(self, other, operation, option, intersections)
    }
}

/// Boolean operation between NURBS curve and compound curve.
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a CompoundCurve<T, U3>> for NurbsCurve<T, U3> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a CompoundCurve<T, U3>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = other.find_intersection(self, option.clone())?;
        let mut sorted = intersections
            .into_iter()
            .sorted_by(|x, y| {
                x.b_parameter()
                    .partial_cmp(&y.b_parameter())
                    .unwrap_or(Ordering::Equal)
            })
            .collect_vec();
        sorted.iter_mut().for_each(|it| it.swap());
        clip(self, other, operation, option, sorted)
    }
}
