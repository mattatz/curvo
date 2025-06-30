use argmin::core::ArgminFloat;
use nalgebra::Const;

use crate::{
    curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions,
    region::CompoundCurve,
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<CompoundCurve<T, Const<3>>>
    for NurbsCurve<T, Const<3>>
{
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a compound curve is inside a curve.
    fn contains(
        &self,
        other: &CompoundCurve<T, Const<3>>,
        option: Self::Option,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(self.is_closed(), "Curve must be closed");

        let all = other
            .spans()
            .iter()
            .map(|span| self.contains(span, option.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(all.iter().all(|it| *it))
    }
}
