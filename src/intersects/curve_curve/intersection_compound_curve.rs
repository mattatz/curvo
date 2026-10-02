use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{HasIntersection, Intersects},
    region::CompoundCurve,
};

use super::{
    intersection_curve_curve::curve_curve_scales, CompoundCurveIntersection,
    CurveIntersectionSolverOptions,
};

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<CompoundCurveIntersection<'a, T, D>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points with another curve
    #[allow(clippy::type_complexity)]
    fn find_intersection(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        options: Self::Option,
    ) -> Self::Output {
        let res: anyhow::Result<Vec<_>> = self
            .spans()
            .iter()
            .map(|span| {
                span.find_intersection(other, options.clone())
                    .map(|intersections| {
                        intersections
                            .into_iter()
                            .map(|it| CompoundCurveIntersection::new(span, other, it))
                            .collect_vec()
                    })
            })
            .collect();

        let mut res = res?;
        let minimum_distance = options.unwrap_or_default().minimum_distance;

        // An intersection at the joint of two spans is found at the end of one and at the start
        // of the next, and is one intersection.
        (0..res.len()).circular_tuple_windows().for_each(|(a, b)| {
            if a != b {
                let same = match (res[a].last(), res[b].first()) {
                    (Some(ia), Some(ib)) => in_contact_through_joint(ia, ib, minimum_distance),
                    _ => false,
                };
                if same {
                    res[a].pop();
                }
            }
        });

        Ok(res.into_iter().flatten().collect())
    }
}

/// Whether `ia`, an intersection of one span of a compound curve with a curve, and `ib`, one of
/// the next span with it, are the same intersection found on the two sides of the joint of the
/// spans: like two candidates on one curve, they are when the compound curve and the other curve
/// are still in contact halfway between them, closer than `minimum_distance` relative to their
/// size.
fn in_contact_through_joint<T, D>(
    ia: &CompoundCurveIntersection<T, D>,
    ib: &CompoundCurveIntersection<T, D>,
    minimum_distance: T,
) -> bool
where
    T: FloatingPoint,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let (span_a, span_b, other) = (ia.a_curve(), ib.a_curve(), ia.b_curve());
    let scales = curve_curve_scales(span_a, other, minimum_distance);

    // How far each is from the joint, as a fraction of its span. Halfway between them through
    // the joint is that far from one of them, on the span of the one further from the joint.
    let (length_a, length_b) = (
        span_a.knots_domain_interval(),
        span_b.knots_domain_interval(),
    );
    let (ta, tb) = (ia.a().2, ib.a().2);
    let from_a = (span_a.knots_domain().1 - ta) / length_a;
    let from_b = (tb - span_b.knots_domain().0) / length_b;
    let half = (from_a + from_b) * T::from_f64(0.5).unwrap();
    let on_compound = if from_a >= from_b {
        span_a.point_at(ta + half * length_a)
    } else {
        span_b.point_at(tb - half * length_b)
    };
    let on_other = other.point_at(scales.halfway(1, ia.b().2, ib.b().2));

    (on_compound - on_other).norm() < scales.minimum_distance()
}

impl<'a, T, D> Intersects<'a, &'a CompoundCurve<T, D>> for CompoundCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<CompoundCurveIntersection<'a, T, D>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points with another compound curve
    #[allow(clippy::type_complexity)]
    fn find_intersection(
        &'a self,
        other: &'a CompoundCurve<T, D>,
        options: Self::Option,
    ) -> Self::Output {
        let res: anyhow::Result<Vec<_>> = other
            .spans()
            .iter()
            .map(|span| self.find_intersection(span.curve(), options.clone()))
            .collect();

        Ok(res?.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use nalgebra::{Point2, Vector2, U3};
    use std::f64::consts::{PI, TAU};

    const OPTIONS: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
        minimum_distance: 1e-4,
        knot_domain_division: 500,
        max_iters: 1000,
        step_size_tolerance: 1e-8,
        cost_tolerance: 1e-10,
    };

    fn contains(
        intersections: &[CompoundCurveIntersection<f64, U3>],
        points: &[Point2<f64>],
        epsilon: f64,
    ) -> bool {
        intersections.iter().all(|it| {
            points
                .iter()
                .any(|pt| (it.a().1 - pt).norm() < epsilon || (it.b().1 - pt).norm() < epsilon)
        })
    }

    #[test]
    fn compound_x_curve_intersection() {
        let o = Point2::origin();
        let dx = Vector2::x();
        let dy = Vector2::y();
        let compound_circle = CompoundCurve::try_new(vec![
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
        ])
        .unwrap();
        let rectangle = NurbsCurve2D::polyline(
            &[
                Point2::new(0., 2.),
                Point2::new(0., -2.),
                Point2::new(2., -2.),
                Point2::new(2., 2.),
                Point2::new(0., 2.),
            ],
            true,
        );
        let intersections = compound_circle
            .find_intersection(&rectangle, Some(OPTIONS))
            .unwrap();
        assert_eq!(intersections.len(), 2);
        assert!(contains(
            &intersections,
            &[Point2::new(0., 1.), Point2::new(0., -1.)],
            1e-2
        ));

        let square = NurbsCurve2D::polyline(
            &[
                Point2::new(-1., 1.),
                Point2::new(-1., -1.),
                Point2::new(1., -1.),
                Point2::new(1., 1.),
                Point2::new(-1., 1.),
            ],
            true,
        );
        let intersections = compound_circle
            .find_intersection(&square, Some(OPTIONS))
            .unwrap();
        assert_eq!(intersections.len(), 4);
        assert!(contains(
            &intersections,
            &[
                Point2::new(1., 0.),
                Point2::new(0., 1.),
                Point2::new(-1., 0.),
                Point2::new(0., -1.)
            ],
            1e-2
        ));
    }

    #[test]
    fn compound_x_compound_intersection() {
        let o = Point2::origin();
        let dx = Vector2::x();
        let dy = Vector2::y();
        let compound_circle = CompoundCurve::try_new(vec![
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
        ])
        .unwrap();
        let compound_rectangle = CompoundCurve::try_new(vec![
            NurbsCurve2D::polyline(
                &[
                    Point2::new(-2., -0.5),
                    Point2::new(2., -0.5),
                    Point2::new(2., 0.5),
                ],
                true,
            ),
            NurbsCurve2D::polyline(
                &[
                    Point2::new(2., 0.5),
                    Point2::new(-2., 0.5),
                    Point2::new(-2., -0.5),
                ],
                true,
            ),
        ])
        .unwrap();

        let intersections = compound_circle
            .find_intersection(&compound_rectangle, Some(OPTIONS))
            .unwrap();
        assert_eq!(intersections.len(), 4);
    }

    #[test]
    fn intersections_at_the_joints_do_not_depend_on_the_domain_of_the_spans() {
        let o = Point2::origin();
        let dx = Vector2::x();
        let dy = Vector2::y();
        // the same curve over a domain `factor` times as long
        let over = |curve: NurbsCurve2D<f64>, factor: f64| {
            let knots = curve.knots().iter().map(|k| k * factor).collect();
            NurbsCurve2D::try_new(curve.degree(), curve.control_points().clone(), knots).unwrap()
        };
        // touches the circle at the two joints of its halves and in the middle of each
        let square = NurbsCurve2D::polyline(
            &[
                Point2::new(-1., 1.),
                Point2::new(-1., -1.),
                Point2::new(1., -1.),
                Point2::new(1., 1.),
                Point2::new(-1., 1.),
            ],
            true,
        );
        // crosses each half once, far from the joints
        let line = NurbsCurve2D::polyline(&[Point2::new(-0.3, -2.), Point2::new(0.4, 2.)], false);
        for factor in [1e-3, 1., 1e3] {
            let circle = CompoundCurve::try_new(vec![
                over(
                    NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
                    factor,
                ),
                over(
                    NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
                    factor,
                ),
            ])
            .unwrap();
            // the bounding box trees are divided at random, so this is run again and again
            for _ in 0..10 {
                let touching = circle.find_intersection(&square, None).unwrap();
                assert_eq!(touching.len(), 4, "over a domain x{factor}");
                let crossing = circle.find_intersection(&line, None).unwrap();
                assert_eq!(crossing.len(), 2, "over a domain x{factor}");
            }
        }
    }
}
