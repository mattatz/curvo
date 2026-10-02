use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Point2,
    Vector2, U1,
};

use crate::{
    curve::NurbsCurve,
    intersects::solve::{closest_in_groups, curve_scale, Scales},
    knot::KnotVector,
    misc::{find_line_string_intersection, to_line_string_helper, FloatingPoint},
    prelude::{BoundingBoxTraversal, CurveBoundingBoxTree, HasIntersection, Intersects},
};

use super::{CurveCurveIntersection, CurveIntersectionProblem, CurveIntersectionSolverOptions};

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for NurbsCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<CurveCurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>>>;
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
        let scales = curve_curve_scales(self, other);
        // relative to the size of the curves
        let minimum_distance = scales.distance(options.minimum_distance);

        if self.degree() == 1 && other.degree() == 1 && D::dim() == 3 {
            // 2d polyline intersection
            let p0 = self
                .dehomogenized_control_points()
                .iter()
                .map(|p| Point2::from_slice(p.coords.as_slice()))
                .collect_vec();
            let p1 = other
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
                    let t0 = find_parameter(&p0, self.knots(), pt, i0)?;
                    let t1 = find_parameter(&p1, other.knots(), pt, i1)?;
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
            // These are exact, so only the same place found twice, at a vertex shared by two
            // segments, is one intersection.
            return Ok(closest_in_groups(
                its,
                |_| T::zero(),
                minimum_distance,
                |it| it.a().1,
                scales.is_closed(0),
                |x, y, _| (&x.a().0 - &y.a().0).norm() < minimum_distance,
            ));
        }

        let ta = CurveBoundingBoxTree::new(
            self,
            Some(
                self.knots_domain_interval() / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );
        let tb = CurveBoundingBoxTree::new(
            other,
            Some(
                other.knots_domain_interval()
                    / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );

        let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;

        let candidates = traversed.into_pairs_iter().filter_map(|(a, b)| {
            let ca = a.curve_owned();
            let cb = b.curve_owned();

            let problem = CurveIntersectionProblem::new(&ca, &cb);

            // Define initial parameter vector
            let init_param = Vector2::<T>::new(ca.knots_domain().0, cb.knots_domain().0);

            // Run solver
            let param = scales.solve(problem, &options, init_param)?;

            // An intersection at the end of a domain is found a hair inside it or a hair
            // outside, so the parameters are clamped rather than refused: how far apart the
            // points there are decides. What was found past the end of a curve is where the
            // curves would meet if that one went on, so the candidate is its end and the
            // closest point of the other curve.
            let ta = self.knots().clamp(self.degree(), param[0]);
            let tb = other.knots().clamp(other.degree(), param[1]);
            let (ta, tb) = match (ta != param[0], tb != param[1]) {
                (true, false) => {
                    let closest = other.find_closest_parameter(&self.point_at(ta));
                    (ta, closest.unwrap_or(tb))
                }
                (false, true) => {
                    let closest = self.find_closest_parameter(&other.point_at(tb));
                    (closest.unwrap_or(ta), tb)
                }
                _ => (ta, tb),
            };
            let p0 = self.point_at(ta);
            let p1 = other.point_at(tb);
            Some(CurveCurveIntersection::new((p0, ta), (p1, tb)))
        });

        // Two candidates are one intersection when the curves are still in contact halfway
        // between them. That is so of an intersection found from several pairs of leaves, and of
        // the candidates found all along the stretch where two curves touch and stay closer
        // than the minimum distance.
        Ok(closest_in_groups(
            candidates,
            |it| (&it.a().0 - &it.b().0).norm(),
            minimum_distance,
            |it| it.a().1,
            scales.is_closed(0),
            |x, y, across| {
                let ta = if across {
                    scales.halfway_across(0, x.a().1, y.a().1)
                } else {
                    (x.a().1 + y.a().1) * T::from_f64(0.5).unwrap()
                };
                let tb = scales.halfway(1, x.b().1, y.b().1);
                (self.point_at(ta) - other.point_at(tb)).norm() < minimum_distance
            },
        ))
    }
}

/// The scales of the intersection of two curves.
fn curve_curve_scales<T, D>(a: &NurbsCurve<T, D>, b: &NurbsCurve<T, D>) -> Scales<T, 2>
where
    T: FloatingPoint,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let (a_size, a_parameter) = curve_scale(a);
    let (b_size, b_parameter) = curve_scale(b);
    Scales::new(&[a_size, b_size], [a_parameter, b_parameter])
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

    /// The bounding box trees are divided at random, so a case that is only found some of the
    /// time is run again and again.
    const RUNS: usize = 20;

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
        for _ in 0..RUNS {
            let intersections = arc.find_intersection(&bar, None).unwrap();
            assert_eq!(intersections.len(), 1);
            assert!((intersections[0].a().1 - end).abs() < 1e-6);
            assert!((intersections[0].a().0 - Point2::new(5., 2.)).norm() < 1e-5);
        }
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
            for _ in 0..RUNS {
                let intersections = a.find_intersection(&b, None).unwrap();
                assert_eq!(intersections.len(), 7, "scaled by {scale}");
            }
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
        for _ in 0..RUNS {
            assert_eq!(unit.find_intersection(&tangent, None).unwrap().len(), 1);
            assert_eq!(unit.find_intersection(&beside, None).unwrap().len(), 1);
            assert_eq!(half.find_intersection(&square, None).unwrap().len(), 3);
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
        for _ in 0..RUNS {
            let intersections = upper.find_intersection(&lower, None).unwrap();
            assert_eq!(intersections.len(), 2);
            assert!((intersections[0].a().0 - Point2::new(0., 0.)).norm() < 1e-5);
            assert!((intersections[1].a().0 - Point2::new(3., 0.)).norm() < 1e-5);
        }
    }
}
