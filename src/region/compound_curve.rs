use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OMatrix,
    OPoint, U1,
};
use num_traits::Float;

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Invertible, Transformable},
    prelude::{
        compound_curve_intersection::CompoundCurveIntersection, CurveIntersectionSolverOptions,
        HasIntersection,
    },
};

use super::curve_direction::CurveDirection;

/// A struct representing a compound curve.
#[derive(Clone, Debug)]
pub struct CompoundCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    spans: Vec<NurbsCurve<T, D>>,
}

impl<T: FloatingPoint, D: DimName> CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(spans: Vec<NurbsCurve<T, D>>) -> Self
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let eps = T::from_f64(1e-5).unwrap();

        // Ensure the adjacent spans are connected in the forward direction.
        let mut curves = spans.clone();
        let mut connected = vec![curves.remove(0)];
        while !curves.is_empty() {
            let current = connected.len() - 1;
            let last = &connected[current];
            let found = curves.iter().enumerate().find_map(|(i, c)| {
                CurveDirection::new(last, c, eps).map(|direction| (i, direction))
            });
            match found {
                Some((index, direction)) => {
                    let next = curves.remove(index);
                    match direction {
                        CurveDirection::Forward => {
                            connected.push(next);
                        }
                        CurveDirection::Backward => {
                            connected.insert(current, next);
                        }
                        CurveDirection::Facing => {
                            connected.push(next.inverse());
                        }
                        CurveDirection::Opposite => {
                            if current == 0 {
                                connected.insert(current, next.inverse());
                            } else {
                                println!("Cannot handle opposite direction");
                            }
                        }
                    }
                }
                None => {
                    println!("No connection found");
                    break;
                }
            }
        }

        Self { spans: connected }
    }

    pub fn spans(&self) -> &[NurbsCurve<T, D>] {
        &self.spans
    }

    pub fn spans_mut(&mut self) -> &mut [NurbsCurve<T, D>] {
        &mut self.spans
    }

    /// Returns the total length of the compound curve.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{PI, TAU};
    /// use approx::{assert_relative_eq};
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
    /// let length = compound.try_length().unwrap();
    /// assert_relative_eq!(length, TAU);
    /// ```
    pub fn try_length(&self) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let lengthes: anyhow::Result<Vec<T>> =
            self.spans.iter().map(|span| span.try_length()).collect();
        let total = lengthes?.iter().fold(T::zero(), |a, b| a + *b);
        Ok(total)
    }

    /// Find the closest point on the curve to a given point
    /// # Example
    /// ```
    /// use nalgebra::{Point2, Vector2};
    /// use curvo::prelude::*;
    /// use std::f64::consts::{PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
    /// assert_relative_eq!(compound.find_closest_point(&Point2::new(3.0, 0.0)).unwrap(), Point2::new(1., 0.));
    /// ```
    pub fn find_closest_point(
        &self,
        point: &OPoint<T, DimNameDiff<D, U1>>,
    ) -> anyhow::Result<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let res: anyhow::Result<Vec<_>> = self
            .spans
            .iter()
            .map(|span| span.find_closest_point(point))
            .collect();
        let res = res?;
        let closest = res
            .into_iter()
            .map(|pt| {
                let delta = &pt - point;
                let distance = delta.norm_squared();
                (pt, distance)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        match closest {
            Some(closest) => Ok(closest.0),
            _ => Err(anyhow::anyhow!("Failed to find the closest point")),
        }
    }

    /// Find the intersection points with another curve
    #[allow(clippy::type_complexity)]
    pub fn find_intersections<'a>(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        options: Option<CurveIntersectionSolverOptions<T>>,
    ) -> anyhow::Result<Vec<CompoundCurveIntersection<'a, T, D>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let res: anyhow::Result<Vec<_>> = self
            .spans
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

        /*
        res.iter().for_each(|it| {
            let pts = it.iter().map(|i| &i.a().1).collect_vec();
            println!("{:?}", pts);
        });
        */

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

impl<T: FloatingPoint, D: DimName> FromIterator<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = NurbsCurve<T, D>>>(iter: I) -> Self {
        Self {
            spans: iter.into_iter().collect(),
        }
    }
}

impl<T: FloatingPoint, D: DimName> From<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: NurbsCurve<T, D>) -> Self {
        Self::new(vec![value])
    }
}

impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for CompoundCurve<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.spans
            .iter_mut()
            .for_each(|span| span.transform(transform));
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn invert(&mut self) {
        self.spans.iter_mut().for_each(|span| span.invert());
        self.spans.reverse();
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{PI, TAU};

    use nalgebra::{Point2, Vector2, U3};

    use crate::prelude::*;

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
                .any(|pt| (it.a().1 - pt).norm() < epsilon || (&it.b().1 - pt).norm() < epsilon)
        })
    }

    #[test]
    fn compound_x_curve_intersection() {
        let o = Point2::origin();
        let dx = Vector2::x();
        let dy = Vector2::y();
        let compound = CompoundCurve::new(vec![
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
            NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
        ]);
        let rectangle = NurbsCurve2D::polyline(&[
            Point2::new(0., 2.),
            Point2::new(0., -2.),
            Point2::new(2., -2.),
            Point2::new(2., 2.),
            Point2::new(0., 2.),
        ]);
        let intersections = compound
            .find_intersections(&rectangle, Some(OPTIONS))
            .unwrap();
        assert_eq!(intersections.len(), 2);
        assert!(contains(
            &intersections,
            &[Point2::new(0., 1.), Point2::new(0., -1.)],
            1e-2
        ));

        let square = NurbsCurve2D::polyline(&[
            Point2::new(-1., 1.),
            Point2::new(-1., -1.),
            Point2::new(1., -1.),
            Point2::new(1., 1.),
            Point2::new(-1., 1.),
        ]);
        let intersections = compound.find_intersections(&square, Some(OPTIONS)).unwrap();
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
}
