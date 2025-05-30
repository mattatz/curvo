use argmin::core::ArgminFloat;
use nalgebra::{Const, Point2};

use crate::{
    contains::contains_curve::x_ray_intersection,
    misc::FloatingPoint,
    prelude::{BoundingBox, CurveIntersectionSolverOptions},
    region::CompoundCurve,
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<T, Const<2>> for CompoundCurve<T, Const<3>> {
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

        let bb: BoundingBox<T, Const<2>> = self.into();
        if !bb.contains(point) {
            return Ok(false);
        }

        let on_boundary = self
            .find_closest_point(point)
            .map(|closest| {
                let delta = closest - point;
                let distance = delta.norm();
                distance
                    < option
                        .as_ref()
                        .map(|opt| opt.minimum_distance)
                        .unwrap_or(T::from_f64(1e-6).unwrap())
            })
            .unwrap_or(false);
        if on_boundary {
            return Ok(true);
        }

        let size = bb.size();
        let sx = size.x * T::from_f64(2.).unwrap();
        let intersections: anyhow::Result<Vec<_>> = self
            .spans()
            .iter()
            .map(|span| x_ray_intersection(span, point, sx, option.clone()))
            .collect();

        let count = intersections?.iter().map(|its| its.len()).sum::<usize>();
        Ok(count % 2 == 1)
    }
}
