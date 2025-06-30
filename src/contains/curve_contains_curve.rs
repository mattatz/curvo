use argmin::core::ArgminFloat;
use nalgebra::Const;

use crate::{
    curve::NurbsCurve,
    misc::{to_line_string_helper, EndPoints, FloatingPoint},
    prelude::CurveIntersectionSolverOptions,
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<NurbsCurve<T, Const<3>>> for NurbsCurve<T, Const<3>> {
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a curve is inside another curve.
    fn contains(
        &self,
        other: &NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> anyhow::Result<bool> {
        anyhow::ensure!(self.is_closed(), "Curve must be closed");

        match (self.degree(), other.degree()) {
            (1, 1) => {
                // use robust contains method
                let l0 = to_line_string_helper(&self.dehomogenized_control_points());
                let l1 = to_line_string_helper(&other.dehomogenized_control_points());
                Ok(geo::Contains::contains(&l0, &l1))
            }
            _ => {
                let pt = other.first_point();
                self.contains(&pt, option.clone())
            }
        }
    }
}
