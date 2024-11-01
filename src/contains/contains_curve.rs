use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{Const, Point2, Vector2, U3};
use num_traits::Float;

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{BoundingBox, BoundingBoxTraversal, CurveIntersectionSolverOptions},
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<T, Const<2>> for NurbsCurve<T, Const<3>> {
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a point is inside a closed curve by ray casting method.
    /// # Example
    /// ```
    /// use nalgebra::{Point2, Vector2};
    /// use curvo::prelude::*;
    /// let circle = NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap();
    /// assert!(circle.contains(&Point2::new(-0.2, 0.2), None).unwrap());
    /// assert!(!circle.contains(&Point2::new(2., 0.), None).unwrap());
    /// assert!(!circle.contains(&Point2::new(0., 1.1), None).unwrap());
    /// ```
    fn contains(&self, point: &Point2<T>, option: Self::Option) -> anyhow::Result<bool> {
        anyhow::ensure!(self.is_closed(), "Curve must be closed");

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

        // curve & ray intersections
        match x_ray_intersection(self, point, size.x * T::from_f64(2.).unwrap(), option) {
            Ok(its) => Ok(its.len() % 2 == 1),
            Err(e) => Err(e),
        }
    }
}

pub fn x_ray_intersection<T: FloatingPoint + ArgminFloat>(
    curve: &NurbsCurve<T, U3>,
    point: &Point2<T>,
    ray_length: T,
    option: Option<CurveIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<Point2<T>>> {
    let option = option.unwrap_or_default();
    let ray = NurbsCurve::polyline(&[*point, point + Vector2::x() * ray_length], true);
    let traversed = BoundingBoxTraversal::try_traverse(
        curve,
        &ray,
        Some(curve.knots_domain_interval() / T::from_usize(option.knot_domain_division).unwrap()),
        Some(ray.knots_domain_interval() / T::from_usize(1).unwrap()),
    )?;

    let mut intersections = traversed
        .pairs()
        .iter()
        .filter_map(|(item, _)| {
            let curve = item.curve();
            let (start, end) = curve.knots_domain();
            let p_start = curve.point_at(start);
            let p_end = curve.point_at(end);

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

                if Float::abs(dy) < option.minimum_distance {
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

    let eps = option.minimum_distance;
    let filtered = intersections
        .into_iter()
        .coalesce(|x, y| {
            // merge intersections that are close
            let dt = x.0 - y.0;
            if dt.norm() < eps {
                Ok(x)
            } else {
                Err((x, y))
            }
        })
        .map(|p| p.0)
        .collect_vec();

    Ok(filtered)
}
