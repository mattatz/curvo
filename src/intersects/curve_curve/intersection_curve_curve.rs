use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Point2,
    Vector2, U1,
};

use crate::{
    curve::NurbsCurve,
    intersects::solve::{closest_to_an_end, curve_scale, group_in_order, Scales},
    knot::KnotVector,
    misc::{find_line_string_intersection, to_line_string_helper, FloatingPoint},
    prelude::{BoundingBoxTraversal, CurveBoundingBoxTree, HasIntersection, Intersects},
};

use super::{CurveCurveIntersection, CurveIntersectionProblem, CurveIntersectionSolverOptions};

/// The intersection of two curves of dimension `D`, homogeneous.
type Intersection<T, D> = CurveCurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>;

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for NurbsCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<Intersection<T, D>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points with another curve by gauss-newton line search
    /// * `other` - The other curve to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Point3, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let line = NurbsCurve2D::try_new(
    ///     1,
    ///     vec![
    ///         Point3::new(-2.0, 0.0, 1.),
    ///         Point3::new(2.0, 0.0, 1.),
    ///     ],
    ///     vec![0., 0., 1., 1.],
    /// ).unwrap();
    ///
    /// // Hyperparameters for the intersection solver
    /// let options = CurveIntersectionSolverOptions {
    ///     minimum_distance: 1e-5, // distance below which two points are an intersection, relative to the size of the curves
    ///     cost_tolerance: 1e-12, // cost tolerance for the solver convergence
    ///     max_iters: 200, // maximum number of iterations in the solver
    ///     ..Default::default()
    /// };
    ///
    /// let mut intersections = unit_circle.find_intersection(&line, Some(options)).unwrap();
    /// assert_eq!(intersections.len(), 2);
    ///
    /// intersections.sort_by(|i0, i1| {
    ///     i0.a().0.x.partial_cmp(&i1.a().0.x).unwrap()
    /// });
    /// let p0 = &intersections[0];
    /// assert_relative_eq!(p0.a().0, Point2::new(-1.0, 0.0), epsilon = 1e-5);
    /// let p1 = &intersections[1];
    /// assert_relative_eq!(p1.a().0, Point2::new(1.0, 0.0), epsilon = 1e-5);
    /// ```
    fn find_intersection(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        option: Self::Option,
    ) -> Self::Output {
        let options = option.unwrap_or_default();
        let scales = curve_curve_scales(self, other, options.minimum_distance);

        if self.degree() == 1 && other.degree() == 1 && D::dim() == 3 {
            return polyline_intersections(self, other, scales.minimum_distance());
        }

        let ta = CurveBoundingBoxTree::with_divisions(self, options.knot_domain_division);
        let tb = CurveBoundingBoxTree::with_divisions(other, options.knot_domain_division);
        let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;

        let candidates = traversed.into_pairs_iter().flat_map(|(a, b)| {
            let (ca, cb) = (a.curve_owned(), b.curve_owned());
            let problem = CurveIntersectionProblem::new(&ca, &cb);
            let leaf = [ca.knots_domain(), cb.knots_domain()];
            let found = scales.solve_leaf(&problem, &options, leaf);
            found.into_iter().flatten().map(|(param, past)| {
                let (ta, tb) = closest_to_an_end(
                    (param[0], past[0]),
                    (param[1], past[1]),
                    |tb| self.find_closest_parameter(&other.point_at(*tb)).ok(),
                    |ta| other.find_closest_parameter(&self.point_at(*ta)).ok(),
                );
                Vector2::new(ta, tb)
            })
        });

        let distance_at =
            |param: &Vector2<T>| (self.point_at(param[0]) - other.point_at(param[1])).norm();
        let intersections = scales
            .intersections(candidates, 0, distance_at)
            .into_iter()
            .map(|param| {
                let (ta, tb) = (param[0], param[1]);
                CurveCurveIntersection::new((self.point_at(ta), ta), (other.point_at(tb), tb))
            })
            .collect();
        Ok(intersections)
    }
}

/// The intersections of two polylines in 2D, found exactly. Only the same place found twice, at
/// a vertex shared by two segments or at the two ends of a closed polyline, is one intersection.
fn polyline_intersections<T, D>(
    a: &NurbsCurve<T, D>,
    b: &NurbsCurve<T, D>,
    minimum_distance: T,
) -> anyhow::Result<Vec<Intersection<T, D>>>
where
    T: FloatingPoint,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let p0 = a
        .dehomogenized_control_points()
        .iter()
        .map(|p| Point2::from_slice(p.coords.as_slice()))
        .collect_vec();
    let p1 = b
        .dehomogenized_control_points()
        .iter()
        .map(|p| Point2::from_slice(p.coords.as_slice()))
        .collect_vec();
    let l0 = to_line_string_helper(&p0);
    let l1 = to_line_string_helper(&p1);
    let intersections = find_line_string_intersection(&l0, &l1)?;

    let find_parameter = |points: &Vec<Point2<T>>,
                          knots: &KnotVector<T>,
                          point: Point2<f64>,
                          index: usize|
     -> anyhow::Result<T> {
        anyhow::ensure!(index + 1 < points.len(), "index out of bounds");

        let prev = points[index];
        let next = points[index + 1];
        let k0 = knots[index + 1];
        let k1 = knots[index + 2];

        let d = (next - prev).norm();
        let d2 = (next - point.map(|x| T::from_f64(x).unwrap())).norm();
        let t = T::one() - d2 / d;

        Ok(k0 + t * (k1 - k0))
    };

    let its = intersections
        .into_iter()
        .map(|it| {
            let pt = it.point();
            let (i0, i1) = it.line_index();
            let t0 = find_parameter(&p0, a.knots(), pt, i0)?;
            let t1 = find_parameter(&p1, b.knots(), pt, i1)?;
            let pt = OPoint::<T, DimNameDiff<D, U1>>::from_slice(
                &pt.coords
                    .as_slice()
                    .iter()
                    .map(|x| T::from_f64(*x).unwrap())
                    .collect_vec(),
            );
            Ok(CurveCurveIntersection::new((pt.clone(), t0), (pt, t1)))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let same_place = |x: &Intersection<T, D>, y: &Intersection<T, D>, _| {
        (&x.a().0 - &y.a().0).norm() < minimum_distance
    };
    Ok(group_in_order(its, |it| it.a().1, true, same_place)
        .into_iter()
        .filter_map(|group| group.into_iter().next())
        .collect())
}

/// The scales of the intersection of two curves, where two points closer than
/// `minimum_distance`, relative to their size, are an intersection.
pub(super) fn curve_curve_scales<T, D>(
    a: &NurbsCurve<T, D>,
    b: &NurbsCurve<T, D>,
    minimum_distance: T,
) -> Scales<T, 2>
where
    T: FloatingPoint,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let (a_size, a_parameter) = curve_scale(a);
    let (b_size, b_parameter) = curve_scale(b);
    Scales::new(
        &[a_size, b_size],
        [a_parameter, b_parameter],
        minimum_distance,
    )
}

#[cfg(test)]
mod tests {
    use nalgebra::Matrix3;

    use crate::{curve::NurbsCurve2D, interpolation::Interpolation, misc::Transformable};

    use super::*;

    #[test]
    fn test_polyline_x_polyline_intersection_on_edge() {
        let p0 = NurbsCurve2D::<f64>::polyline(
            &[
                Point2::new(0., 0.),
                Point2::new(1., 0.),
                Point2::new(1., 1.),
                Point2::new(0., 1.),
            ],
            false,
        );
        let p1 = NurbsCurve2D::<f64>::polyline(&[Point2::new(1., -1.), Point2::new(1., 2.)], false);

        let intersections = p0.find_intersection(&p1, None).unwrap();
        assert_eq!(intersections.len(), 2);
    }

    #[test]
    fn an_intersection_at_the_end_of_a_curve_is_found() {
        // the end of the arc touches the middle of the bar
        let arc = NurbsCurve2D::<f64>::interpolate(
            &vec![
                Point2::new(0., 0.),
                Point2::new(1., 2.),
                Point2::new(3., 3.),
                Point2::new(5., 2.),
            ],
            3,
        )
        .unwrap();
        let bar = NurbsCurve2D::<f64>::interpolate(
            &vec![
                Point2::new(5., -1.),
                Point2::new(5.2, 1.),
                Point2::new(5., 2.),
                Point2::new(4.6, 4.),
                Point2::new(5., 6.),
            ],
            3,
        )
        .unwrap();
        let (_, end) = arc.knots_domain();
        let intersections = arc.find_intersection(&bar, None).unwrap();
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].a().1 - end).abs() < 1e-6);
        assert!((intersections[0].a().0 - Point2::new(5., 2.)).norm() < 1e-5);
    }

    fn circle(center: Point2<f64>) -> NurbsCurve2D<f64> {
        NurbsCurve2D::try_circle(&center, &Vector2::x(), &Vector2::y(), 1.).unwrap()
    }

    #[test]
    fn intersections_do_not_depend_on_the_scale() {
        let wave = |phase: f64| {
            let points = (0..32)
                .map(|i| {
                    let x = i as f64;
                    Point2::new(x, (x * 0.7 + phase).sin() * 3.)
                })
                .collect_vec();
            NurbsCurve2D::<f64>::interpolate(&points, 3).unwrap()
        };
        let (a, b) = (wave(0.), wave(1.3));
        for scale in [1., 1e-3, 1e3] {
            let scaling = Matrix3::new_scaling(scale);
            let (a, b) = (a.transformed(&scaling), b.transformed(&scaling));
            let intersections = a.find_intersection(&b, None).unwrap();
            assert_eq!(intersections.len(), 7, "scaled by {scale}");
        }
    }

    #[test]
    fn curves_that_touch_intersect_once_where_they_touch() {
        let unit = circle(Point2::origin());
        let tangent =
            NurbsCurve2D::<f64>::polyline(&[Point2::new(-2., 1.), Point2::new(2., 1.)], false);
        // at the seam of the first circle
        let beside = circle(Point2::new(2., 0.));
        // the half circle touches the square at its two ends and at its top
        let half = NurbsCurve2D::<f64>::try_arc(
            &Point2::origin(),
            &Vector2::x(),
            &Vector2::y(),
            1.,
            0.,
            std::f64::consts::PI,
        )
        .unwrap();
        let square = NurbsCurve2D::<f64>::polyline(
            &[
                Point2::new(-1., 1.),
                Point2::new(-1., -1.),
                Point2::new(1., -1.),
                Point2::new(1., 1.),
                Point2::new(-1., 1.),
            ],
            true,
        );
        assert_eq!(unit.find_intersection(&tangent, None).unwrap().len(), 1);
        assert_eq!(unit.find_intersection(&beside, None).unwrap().len(), 1);
        assert_eq!(half.find_intersection(&square, None).unwrap().len(), 3);
    }

    #[test]
    fn two_intersections_next_to_each_other_are_both_found() {
        // The line crosses the circle twice just under its top, 0.09 apart. The solver goes from
        // most leaves around there to the same one of the two.
        let unit = circle(Point2::origin());
        let under = NurbsCurve2D::<f64>::polyline(
            &[Point2::new(-2., 0.999), Point2::new(2., 0.999)],
            false,
        );
        // at x = -0.0447 and x = 0.0447: where the line crosses so shallowly, the two curves are
        // within the minimum distance of each other over a longer stretch than they are apart
        let x = (1f64 - 0.999 * 0.999).sqrt();
        let mut intersections = unit.find_intersection(&under, None).unwrap();
        intersections.sort_by(|i, j| i.a().0.x.partial_cmp(&j.a().0.x).unwrap());
        assert_eq!(intersections.len(), 2);
        for (intersection, x) in intersections.iter().zip([-x, x]) {
            let (on_circle, on_line) = (intersection.a().0, intersection.b().0);
            assert!((on_circle - on_line).norm() < 1e-4);
            assert!((on_circle.x - x).abs() < 5e-3);
        }
    }

    #[test]
    fn the_same_curves_have_the_same_intersections_every_time() {
        let wave = |phase: f64| {
            let points = (0..32)
                .map(|i| {
                    let x = i as f64;
                    Point2::new(x, (x * 0.7 + phase).sin() * 3.)
                })
                .collect_vec();
            NurbsCurve2D::<f64>::interpolate(&points, 3).unwrap()
        };
        let (a, b) = (wave(0.), wave(1.3));
        let parameters = || {
            let intersections = a.find_intersection(&b, None).unwrap();
            intersections
                .iter()
                .map(|it| (it.a().1, it.b().1))
                .collect_vec()
        };
        let first = parameters();
        assert_eq!(first.len(), 7);
        for _ in 0..3 {
            assert_eq!(parameters(), first);
        }
    }

    #[test]
    fn intersections_at_both_ends_of_two_curves_are_found() {
        let lens = |y: f64| {
            NurbsCurve2D::<f64>::interpolate(
                &vec![
                    Point2::new(0., 0.),
                    Point2::new(1., y),
                    Point2::new(2., y * 1.2),
                    Point2::new(3., 0.),
                ],
                3,
            )
            .unwrap()
        };
        let (upper, lower) = (lens(1.), lens(-1.));
        let intersections = upper.find_intersection(&lower, None).unwrap();
        assert_eq!(intersections.len(), 2);
        assert!((intersections[0].a().0 - Point2::new(0., 0.)).norm() < 1e-5);
        assert!((intersections[1].a().0 - Point2::new(3., 0.)).norm() < 1e-5);
    }
}
