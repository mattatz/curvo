use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

use super::Split;

impl<T: FloatingPoint, D: DimName> Split for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Option = T;

    /// Split the curve into two curves before and after the parameter
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::TAU;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let (min, max) = unit_circle.knots_domain();
    /// let u = (min + max) / 2.;
    /// let (left, right) = unit_circle.try_split(u).unwrap();
    /// assert_eq!(left.knots_domain().1, u);
    /// assert_eq!(right.knots_domain().0, u);
    /// ```
    fn try_split(&self, u: T) -> anyhow::Result<(Self, Self)> {
        let u = self.knots().clamp(self.degree(), u);
        let knots_to_insert = (0..=self.degree()).map(|_| u).collect_vec();
        let mut cloned = self.clone();
        cloned.try_refine_knot(knots_to_insert)?;

        let n = self.knots().len() - self.degree() - 2;
        let s = self.knots().find_knot_span_index(n, self.degree(), u);
        let knots0 = cloned.knots().as_slice()[0..=(s + self.degree() + 1)].to_vec();
        let knots1 = cloned.knots().as_slice()[s + 1..].to_vec();
        let cpts0 = cloned.control_points()[0..=s].to_vec();
        let cpts1 = cloned.control_points()[s + 1..].to_vec();
        Ok((
            Self::try_new(self.degree(), cpts0, knots0)?,
            Self::try_new(self.degree(), cpts1, knots1)?,
        ))
    }
}
