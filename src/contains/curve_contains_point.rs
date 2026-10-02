use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint, Point2, Vector2, U3};

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{
        BoundingBox, BoundingBoxTraversal, CurveBoundingBoxTree, CurveIntersectionSolverOptions,
    },
};

use super::Contains;

impl<T: FloatingPoint + ArgminFloat> Contains<OPoint<T, Const<2>>> for NurbsCurve<T, Const<3>> {
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a point is inside a closed curve by ray casting method.
    ///
    /// A point within the minimum distance of the curve, relative to its size, is inside.
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

        let options = option.unwrap_or_default();
        is_inside(
            &self.into(),
            point,
            &options,
            || self.find_closest_point(point),
            |ray_length| {
                let crossings = x_ray_intersection(self, point, ray_length, Some(options.clone()))?;
                Ok(crossings.len())
            },
        )
    }
}

/// Whether `point` is inside the closed boundary within `bounding_box`: on it, within the minimum
/// distance of the `closest` point of the boundary, or on the inner side of it, when a ray from
/// the point has an odd number of `crossings` with it.
///
/// The minimum distance is relative to the size of the boundary, the diagonal of its bounding
/// box, and `crossings` is given the length of the ray.
pub(super) fn is_inside<T: FloatingPoint>(
    bounding_box: &BoundingBox<T, Const<2>>,
    point: &Point2<T>,
    options: &CurveIntersectionSolverOptions<T>,
    closest: impl FnOnce() -> anyhow::Result<Point2<T>>,
    crossings: impl FnOnce(T) -> anyhow::Result<usize>,
) -> anyhow::Result<bool> {
    let size = bounding_box.size();
    let tolerance = options.minimum_distance * size.norm();

    // nowhere near the boundary
    let (min, max) = (bounding_box.min(), bounding_box.max());
    if (0..2).any(|i| point[i] < min[i] - tolerance || max[i] + tolerance < point[i]) {
        return Ok(false);
    }

    let on_boundary = closest().is_ok_and(|closest| (closest - point).norm() < tolerance);
    if on_boundary {
        return Ok(true);
    }

    let count = crossings(size.x * T::from_f64(2.).unwrap())?;
    Ok(count % 2 == 1)
}

/// The points where the ray from `point` along x, `ray_length` long, crosses the curve.
///
/// A curve that only touches the ray does not cross it, and one that starts or ends on the ray
/// crosses it when it comes from above or leaves upwards, so that the crossings of a closed
/// curve tell whether the point is inside it.
pub fn x_ray_intersection<T: FloatingPoint + ArgminFloat>(
    curve: &NurbsCurve<T, U3>,
    point: &Point2<T>,
    ray_length: T,
    option: Option<CurveIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<Point2<T>>> {
    let options = option.unwrap_or_default();
    let ray = NurbsCurve::polyline(&[*point, point + Vector2::x() * ray_length], true);
    let ta = CurveBoundingBoxTree::with_divisions(curve, options.knot_domain_division);
    let tb = CurveBoundingBoxTree::with_divisions(&ray, 1);
    let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;

    let above = |t: T, leaf: &NurbsCurve<T, U3>| point.y < leaf.point_at(t).y;
    let crossings = traversed
        .pairs()
        .iter()
        .flat_map(|(leaf, _)| {
            let leaf = leaf.curve();
            let (start, end) = leaf.knots_domain();
            let (above_start, above_end) = (above(start, leaf), above(end, leaf));

            // The ray crosses a stretch whose ends are on its two sides. A leaf with both ends
            // on one side may still go to the other side and come back: it is then two such
            // stretches, on either side of where it goes the furthest.
            let stretches = if above_start != above_end {
                vec![(start, end)]
            } else {
                let furthest = furthest(leaf, (start, end), !above_start, options.max_iters);
                if above(furthest, leaf) != above_start {
                    vec![(start, furthest), (furthest, end)]
                } else {
                    vec![]
                }
            };
            stretches
                .into_iter()
                .map(|stretch| crossing(leaf, stretch, point.y, options.max_iters))
        })
        .filter(|crossing| point.x <= crossing.x)
        .collect();
    Ok(crossings)
}

/// The point where a stretch of the curve, with its ends on the two sides of the line at `y`,
/// crosses it, by bisection.
fn crossing<T: FloatingPoint>(
    curve: &NurbsCurve<T, U3>,
    (mut from, mut to): (T, T),
    y: T,
    max_iters: u64,
) -> Point2<T> {
    let half = T::from_f64(0.5).unwrap();
    let above_from = y < curve.point_at(from).y;
    for _ in 0..max_iters {
        let middle = (from + to) * half;
        if middle <= from || to <= middle {
            break;
        }
        if (y < curve.point_at(middle).y) == above_from {
            from = middle;
        } else {
            to = middle;
        }
    }
    curve.point_at((from + to) * half)
}

/// The parameter in a stretch of the curve where it goes the furthest `up`, or down, by ternary
/// search: the stretch is taken to go one way and come back.
fn furthest<T: FloatingPoint>(
    curve: &NurbsCurve<T, U3>,
    (mut from, mut to): (T, T),
    up: bool,
    max_iters: u64,
) -> T {
    let height = |t: T| {
        let y = curve.point_at(t).y;
        if up {
            y
        } else {
            -y
        }
    };
    let third = T::from_f64(1. / 3.).unwrap();
    for _ in 0..max_iters {
        let (a, b) = (from + (to - from) * third, to - (to - from) * third);
        if !(from < a && a < b && b < to) {
            break;
        }
        if height(a) < height(b) {
            from = a;
        } else {
            to = b;
        }
    }
    (from + to) * T::from_f64(0.5).unwrap()
}
