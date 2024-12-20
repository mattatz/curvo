use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};
use num_traits::Float;

use crate::{curve::NurbsCurve, misc::FloatingPoint, region::CompoundCurve};

use super::{
    CompoundCurveIntersection, CurveIntersectionSolverOptions, HasIntersection, Intersects,
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
    fn find_intersections(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        options: Self::Option,
    ) -> Self::Output {
        let res: anyhow::Result<Vec<_>> = self
            .spans()
            .iter()
            .map(|span| {
                span.find_intersections(other, options.clone())
                    .map(|intersections| {
                        intersections
                            .into_iter()
                            .map(|it| CompoundCurveIntersection::new(span, other, it))
                            .collect_vec()
                    })
            })
            .collect();

        let mut res = res?;
        let eps = T::from_f64(1e-2).unwrap();

        (0..res.len()).circular_tuple_windows().for_each(|(a, b)| {
            if a != b {
                let ia = res[a].last();
                let ib = res[b].first();
                let cull = match (ia, ib) {
                    (Some(ia), Some(ib)) => {
                        // cull the last point in res[a] if it is too close to the first point in res[b]
                        let da = Float::abs(ia.a().0.knots_domain().1 - ia.a().2);
                        let db = Float::abs(ib.a().0.knots_domain().0 - ib.a().2);
                        da < eps && db < eps
                    }
                    _ => false,
                };
                if cull {
                    res[a].pop();
                }
            }
        });

        Ok(res.into_iter().flatten().collect())
    }
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
    fn find_intersections(
        &'a self,
        other: &'a CompoundCurve<T, D>,
        options: Self::Option,
    ) -> Self::Output {
        let res: anyhow::Result<Vec<_>> = other
            .spans()
            .iter()
            .map(|span| self.find_intersections(span, options.clone()))
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
        let compound_circle = CompoundCurve::new(vec![
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
        ]);
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
            .find_intersections(&rectangle, Some(OPTIONS))
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
            .find_intersections(&square, Some(OPTIONS))
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
        let compound_circle = CompoundCurve::new(vec![
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
        ]);
        let compound_rectangle = CompoundCurve::new(vec![
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
        ]);

        let intersections = compound_circle
            .find_intersections(&compound_rectangle, Some(OPTIONS))
            .unwrap();
        assert_eq!(intersections.len(), 4);
    }
}
