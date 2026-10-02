use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

use super::{Split, SplitAt};

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
        let split = SplitAt::new(self.knots(), degree, u);
        let (cpts, knots) = split.refine(self.control_points(), self.knots());
        let (cpts0, cpts1) = split.divide_control_points(cpts);
        let (knots0, knots1) = split.divide_knots(knots);
        Ok((
            Self::try_new(degree, cpts0, knots0)?,
            Self::try_new(degree, cpts1, knots1)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Point4;

    use crate::{curve::NurbsCurve3D, split::Split};

    fn curve(degree: usize, knots: Vec<f64>) -> NurbsCurve3D<f64> {
        let points = (0..knots.len() - degree - 1)
            .map(|i| {
                let x = i as f64;
                let w = 1. + 0.1 * x;
                Point4::new(x * w, x.sin() * w, x.cos() * w, w)
            })
            .collect();
        NurbsCurve3D::try_new(degree, points, knots).unwrap()
    }

    #[test]
    fn the_halves_of_a_split_curve_trace_the_curve() {
        let curves = [
            // clamped, with repeated interior knots
            curve(3, vec![0., 0., 0., 0., 1., 2., 2., 2., 3., 4., 4., 4., 4.]),
            // unclamped
            curve(2, (0..10).map(|i| i as f64).collect()),
            // a single span, of a degree beyond the knots kept on the stack
            curve(17, [vec![0.; 18], vec![1.; 18]].concat()),
        ];
        for curve in curves {
            let (start, end) = curve.knots_domain();
            let mut us: Vec<f64> = (1..10)
                .map(|i| start + (end - start) * i as f64 / 10.)
                .collect();
            // on the interior knots
            us.extend(curve.knots().iter().filter(|k| start < **k && **k < end));
            for u in us {
                let (head, tail) = curve.try_split(u).unwrap();
                assert_eq!(head.knots_domain(), (start, u));
                assert_eq!(tail.knots_domain(), (u, end));
                for (half, (from, to)) in [(head, (start, u)), (tail, (u, end))] {
                    assert_eq!(half.degree(), curve.degree());
                    assert_eq!(
                        half.knots().len(),
                        half.control_points().len() + half.degree() + 1
                    );
                    for i in 0..=8 {
                        let t = from + (to - from) * i as f64 / 8.;
                        assert_relative_eq!(half.point_at(t), curve.point_at(t), epsilon = 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn a_split_outside_the_domain_is_a_split_at_its_end() {
        let curve = curve(3, vec![0., 0., 0., 0., 1., 2., 2., 2., 3., 4., 4., 4., 4.]);
        let (start, end) = curve.knots_domain();
        for (u, at) in [(start - 1., start), (end + 1., end)] {
            let (head, tail) = curve.try_split(u).unwrap();
            let (expected_head, expected_tail) = curve.try_split(at).unwrap();
            assert_eq!(head, expected_head);
            assert_eq!(tail, expected_tail);
        }
    }
}
