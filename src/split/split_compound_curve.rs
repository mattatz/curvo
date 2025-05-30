use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{misc::FloatingPoint, region::CompoundCurve};

use super::Split;

impl<T: FloatingPoint, D: DimName> Split for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Option = T;

    /// Split the compound curve into two compound curves before and after the parameter
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{FRAC_PI_2, PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    ///
    /// let (left, right) = compound.try_split(PI).unwrap();
    /// assert_relative_eq!(left.point_at(left.knots_domain().0), Point2::new(1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(left.point_at(left.knots_domain().1), Point2::new(-1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(right.point_at(right.knots_domain().0), Point2::new(-1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(right.point_at(right.knots_domain().1), Point2::new(1., 0.), epsilon = 1e-10);
    ///
    /// // right is empty, so the split is not possible
    /// assert!(compound.try_split(TAU).is_err());
    ///
    /// let (left, right) = compound.try_split(FRAC_PI_2).unwrap();
    /// assert_relative_eq!(left.point_at(left.knots_domain().0), Point2::new(1., 0.), epsilon = 1e-10);
    /// assert_relative_eq!(left.point_at(left.knots_domain().1 - 1e-10), Point2::new(0., 1.), epsilon = 1e-10);
    /// assert_relative_eq!(right.point_at(right.knots_domain().0), Point2::new(0., 1.), epsilon = 1e-10);
    /// assert_relative_eq!(right.point_at(right.knots_domain().1), Point2::new(1., 0.), epsilon = 1e-10);
    /// ```
    fn try_split(&self, u: T) -> anyhow::Result<(Self, Self)> {
        let index = self.find_span_index(u);
        let (l, r) = self.spans()[index].try_split(u)?;
        let li = l.knots_domain_interval();
        let ri = r.knots_domain_interval();

        let left = self.spans()[0..index].to_vec();
        let right = self.spans()[(index + 1)..].to_vec();

        Ok((
            if li <= T::default_epsilon() {
                anyhow::ensure!(!left.is_empty(), "left is empty");
                Self::new_unchecked(left)
            } else {
                Self::new_unchecked(left.into_iter().chain(vec![l]).collect())
            },
            if ri <= T::default_epsilon() {
                anyhow::ensure!(!right.is_empty(), "right is empty");
                Self::new_unchecked(right)
            } else {
                Self::new_unchecked(vec![r].into_iter().chain(right).collect())
            },
        ))
    }
}
