use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint, Point2};

use crate::{
    contains::curve_contains_point::{is_inside, x_ray_intersection},
    misc::FloatingPoint,
    prelude::CurveIntersectionSolverOptions,
    region::CompoundCurve,
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<OPoint<T, Const<2>>> for CompoundCurve<T, Const<3>> {
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a point is inside a closed curve by ray casting method.
    /// # Example
    /// ```
    /// use nalgebra::{Point2, Vector2};
    /// use curvo::prelude::*;
    /// use std::f64::consts::{PI, TAU};
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    /// assert!(compound.contains(&Point2::new(0.0, 0.0), None).unwrap());
    /// assert!(!compound.contains(&Point2::new(3.0, 0.), None).unwrap());
    /// assert!(compound.contains(&Point2::new(1.0, 0.0), None).unwrap());
    /// ```
    fn contains(&self, point: &Point2<T>, option: Self::Option) -> anyhow::Result<bool> {
        // anyhow::ensure!(self.is_closed(), "Curve must be closed");

        let options = option.unwrap_or_default();
        is_inside(
            &self.into(),
            point,
            &options,
            || self.find_closest_point(point),
            |ray_length| {
                self.spans().iter().try_fold(0, |count, span| {
                    let crossings =
                        x_ray_intersection(span, point, ray_length, Some(options.clone()))?;
                    Ok(count + crossings.len())
                })
            },
        )
    }
}
