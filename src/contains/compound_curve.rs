use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{Const, Point2};

use crate::{
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
    /// let compound = CompoundCurve::new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
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

        let closest = self.find_closest_point(point)?;
        let delta = closest - point;
        let distance = delta.norm();
        if distance
            < option
                .as_ref()
                .map(|opt| opt.minimum_distance)
                .unwrap_or(T::from_f64(1e-6).unwrap())
        {
            return Ok(true);
        }

        /*
        let size = bb.size();
        let dx = Vector2::x();
        let sx = ComplexField::abs(size.dot(&dx));

        // curve & ray intersections
        let ray = NurbsCurve::polyline(&[
            *point,
            point + dx * (ComplexField::abs(delta.x) + sx * T::from_f64(2.).unwrap()),
        ]);

        let option = option.unwrap_or_default();
        let traversed = BoundingBoxTraversal::try_traverse(
            self,
            &ray,
            Some(
                self.knots_domain_interval() / T::from_usize(option.knot_domain_division).unwrap(),
            ),
            Some(ray.knots_domain_interval() / T::from_usize(option.knot_domain_division).unwrap()),
        )?;

        let mut intersections = traversed
            .pairs()
            .iter()
            .filter_map(|(item, _)| {
                let curve = item.curve();
                let (start, end) = curve.knots_domain();
                let p_start = self.point_at(start);
                let p_end = self.point_at(end);

                if p_start.x < point.x && p_end.x < point.x {
                    return None;
                }

                let y_forward = match (point.y < p_start.y, point.y < p_end.y) {
                    (true, true) | (false, false) => {
                        return None;
                    }
                    (false, true) => true,
                    (true, false) => false,
                };

                // binary search for the intersection
                let mut min = start;
                let mut max = end;
                for _i in 0..option.max_iters {
                    let t = (min + max) / T::from_f64(2.).unwrap();
                    let p = curve.point_at(t);
                    let dy = p.y - point.y;

                    if ComplexField::abs(dy) < option.minimum_distance {
                        return Some((p, t));
                    }

                    let over = dy > T::zero();
                    let over = if y_forward { over } else { !over };
                    if over {
                        max = t;
                    } else {
                        min = t;
                    }
                }

                None
            })
            .filter(|(p, _)| point.x <= p.x)
            .collect_vec();

        intersections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let parameter_minimum_distance = T::from_f64(1e-2).unwrap();
        let filtered = intersections
            .iter()
            .coalesce(|x, y| {
                // merge intersections that are close in parameter space
                let dt = y.1 - x.1;
                if dt < parameter_minimum_distance {
                    Ok(x)
                } else {
                    Err((x, y))
                }
            })
            .collect_vec();

        Ok(filtered.len() % 2 == 1)
        */

        todo!();
    }
}
