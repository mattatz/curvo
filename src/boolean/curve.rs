use super::clip::{clip, Clipped};
use super::operation::BooleanOperation;
use super::Boolean;
use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator};

use crate::prelude::{CompoundCurveIntersection, Intersection};
use crate::{curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions};

/// Boolean operation for two NURBS curves.
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for NurbsCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    // type Output = anyhow::Result<Vec<Region<T>>>;
    type Output = anyhow::Result<Clipped<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
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
