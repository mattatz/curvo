use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OMatrix,
    OPoint, U1,
};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Invertible, Transformable},
};

use super::curve_direction::CurveDirection;

/// A struct representing a compound curve.
#[derive(Clone, Debug, PartialEq)]
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
    /// Create a new compound curve from a list of spans.
    /// The spans must be connected.
    pub fn new(spans: Vec<NurbsCurve<T, D>>) -> Self
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        // epsilon for determining the connected points
        let epsilon = T::from_f64(1e-4).unwrap();

        // Ensure the adjacent spans are connected in the forward direction.
        let mut curves = spans.clone();
        let mut connected = vec![curves.remove(0)];
        while !curves.is_empty() {
            let current = connected.len() - 1;
            let last = &connected[current];
            let found = curves.iter().enumerate().find_map(|(i, c)| {
                CurveDirection::new(last, c, epsilon).map(|direction| (i, direction))
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

        // Align knot vectors
        // The first knot vector starts at 0, the rest are aligned to the previous knot vector
        let mut knot_offset = T::zero();
        connected.iter_mut().for_each(|curve| {
            let start = curve.knots().first();
            curve.knots_mut().iter_mut().for_each(|v| {
                *v = *v - start + knot_offset;
            });
            knot_offset = curve.knots().last();
        });

        Self { spans: connected }
    }

    pub fn spans(&self) -> &[NurbsCurve<T, D>] {
        &self.spans
    }

    pub fn spans_mut(&mut self) -> &mut [NurbsCurve<T, D>] {
        &mut self.spans
    }

    /// Evaluate the curve containing the parameter t at the given parameter t.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{FRAC_PI_2, PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
    /// assert_relative_eq!(compound.point_at(0.).unwrap(), Point2::new(1., 0.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(FRAC_PI_2).unwrap(), Point2::new(0., 1.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(PI).unwrap(), Point2::new(-1., 0.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(PI + FRAC_PI_2).unwrap(), Point2::new(0., -1.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(TAU).unwrap(), Point2::new(1., 0.), epsilon = 1e-5);
    /// ```
    pub fn point_at(&self, t: T) -> Option<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let span = self.spans.iter().find(|span| {
            let (d0, d1) = span.knots_domain();
            (d0..=d1).contains(&t)
        });
        span.map(|span| span.point_at(t))
    }

    /// Check if the curve is closed.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{PI, TAU};
    /// use approx::{assert_relative_eq};
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let circle = CompoundCurve::new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
    /// assert!(circle.is_closed());
    /// ```
    pub fn is_closed(&self) -> bool
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let start = self.spans.first().map(|s| s.point_at(s.knots_domain().0));
        let end = self.spans.last().map(|s| s.point_at(s.knots_domain().1));
        let eps = T::default_epsilon() * T::from_usize(10).unwrap();
        match (start, end) {
            (Some(start), Some(end)) => {
                let delta = start - end;
                delta.norm() < eps
            }
            _ => false,
        }
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
