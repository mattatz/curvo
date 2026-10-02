use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{
    curve::{nurbs_curve::refine_knot, NurbsCurve},
    misc::FloatingPoint,
};

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
        let degree = self.degree();
        let u = self.knots().clamp(degree, u);
        // `degree + 1` times `u`, on the stack for the usual degrees
        let (inline, heap);
        let knots_to_insert = if degree < 16 {
            inline = [u; 16];
            &inline[..=degree]
        } else {
            heap = vec![u; degree + 1];
            &heap[..]
        };
        // The refined control points and knots are those of the head followed by those of the
        // tail, so they are divided in place rather than copied out.
        let (mut cpts0, mut knots0) =
            refine_knot(degree, self.control_points(), self.knots(), knots_to_insert);

        let n = self.knots().len() - degree - 2;
        let s = self.knots().find_knot_span_index(n, degree, u);
        let knots1 = knots0[s + 1..].to_vec();
        knots0.truncate(s + degree + 2);
        let cpts1 = cpts0.split_off(s + 1);
        Ok((
            Self::try_new(degree, cpts0, knots0)?,
            Self::try_new(degree, cpts1, knots1)?,
        ))
    }
}
