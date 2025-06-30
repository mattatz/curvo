use argmin::core::ArgminFloat;
use nalgebra::Const;

use crate::{
    misc::{EndPoints, FloatingPoint},
    prelude::CurveIntersectionSolverOptions,
    region::CompoundCurve,
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<CompoundCurve<T, Const<3>>>
    for CompoundCurve<T, Const<3>>
{
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a compound curve is inside a compound curve.
    fn contains(
        &self,
        other: &CompoundCurve<T, Const<3>>,
        option: Self::Option,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(self.is_closed(None), "Compound curve must be closed");

        let pt = other.first_point();
        self.contains(&pt, option.clone())
    }
}
