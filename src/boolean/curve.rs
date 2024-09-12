use super::clip::{clip, Clip};
use super::operation::BooleanOperation;
use super::Boolean;
use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::U3;

use crate::prelude::{CompoundCurveIntersection, Intersection};
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
        let intersections = self.find_intersections(other, option.clone())?;
        let intersections = intersections
            .into_iter()
            .map(|it| CompoundCurveIntersection::new(self, other, it))
            .collect_vec();
        clip(self, other, operation, option, intersections)
    }
}
