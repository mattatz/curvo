use argmin::core::ArgminFloat;
use nalgebra::Const;
use nalgebra::U3;

use crate::prelude::Intersection;
use crate::region::Region;
use crate::{curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions};

use super::clip::Clip;
use super::operation::BooleanOperation;
use super::Boolean;

/// Boolean operation for Region & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, U3>> for Region<T> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        match operation {
            BooleanOperation::Union => {
                let exterior =
                    self.exterior()
                        .boolean(BooleanOperation::Union, other, option.clone())?;
                let interiors = self
                    .interiors()
                    .iter()
                    .map(|interior| {
                        interior.boolean(BooleanOperation::Difference, other, option.clone())
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
            }
            BooleanOperation::Intersection => todo!(),
            BooleanOperation::Difference => todo!(),
        }
        todo!();
        // let intersections = self.find_intersections(other, option.clone())?;
        // clip(self, other, operation, option, intersections)
    }
}
