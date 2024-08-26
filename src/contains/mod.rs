use argmin::core::ArgminFloat;
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, DimName, OPoint, Point2, Vector2,
};

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{BoundingBox, CurveIntersectionSolverOptions},
};

/// Trait for determining if a point is inside a curve.
pub trait Contains<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    type Option;
    fn contains(&self, point: &OPoint<T, D>, option: Self::Option) -> anyhow::Result<bool>;
}

impl<T: FloatingPoint + ArgminFloat> Contains<T, Const<2>> for NurbsCurve<T, Const<3>> {
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Determine if a point is inside a closed curve by ray casting method.
    /// # Example
    /// ```
    /// use nalgebra::{Point2, Vector2};
    /// use curvo::prelude::*;
    /// let circle = NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap();
    /// assert!(circle.contains(&Point2::new(-0.2, 0.2), None).unwrap());
    /// assert!(!circle.contains(&Point2::new(2., 0.), None).unwrap());
    /// assert!(!circle.contains(&Point2::new(0., 1.1), None).unwrap());
    /// ```
    fn contains(&self, point: &Point2<T>, option: Self::Option) -> anyhow::Result<bool> {
        anyhow::ensure!(self.is_closed(), "Curve must be closed");

        let closest = self.find_closest_point(point)?;
        let delta = closest - point;
        if delta.norm()
            < option
                .as_ref()
                .map(|opt| opt.minimum_distance)
                .unwrap_or(T::from_f64(1e-6).unwrap())
        {
            return Ok(true);
        }

        let bb: BoundingBox<T, Const<2>> = self.into();
        let size = bb.size();
        let dx = Vector2::x();
        let sx = ComplexField::abs(size.dot(&dx));

        let ray = NurbsCurve::polyline(&vec![
            point.clone(),
            point + dx * sx * T::from_f64(5.).unwrap(),
        ]);

        let delta = self.knots_domain_interval() * T::from_f64(1e-1).unwrap();
        self.find_intersections(&ray, option).map(|intersections| {
            // filter out the case where the ray passes through a `vertex` of the curve
            let filtered = intersections.into_iter().filter(|it| {
                let parameter = it.a().1;
                let p0 = self.knots().clamp(self.degree(), parameter - delta);
                let p1 = self.knots().clamp(self.degree(), parameter + delta);
                let f0 = point.y < self.point_at(p0).y;
                let f1 = point.y < self.point_at(p1).y;
                f0 != f1
            });
            let count = filtered.count();
            count % 2 == 1
        })

        /*
        self.find_intersections(&ray, option)
            .map(|intersections| intersections.len() % 2 == 1)
        */
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use nalgebra::{Point2, Vector2};

    #[test]
    fn test_circle_boundary_case() {
        let radius = 1.;
        let circle = NurbsCurve2D::<f64>::try_circle(
            &Point2::origin(),
            &Vector2::x(),
            &Vector2::y(),
            radius,
        )
        .unwrap();
        assert!(circle.contains(&Point2::new(radius, 0.0), None).unwrap());
        assert!(circle.contains(&Point2::new(0., radius), None).unwrap());
        assert!(circle.contains(&Point2::new(-radius, 0.), None).unwrap());
        assert!(circle.contains(&Point2::new(0., -radius), None).unwrap());
        assert!(!circle
            .contains(&Point2::new(-radius * 5., radius), None)
            .unwrap());
        assert!(!circle
            .contains(&Point2::new(-radius * 5., -radius), None)
            .unwrap());
    }

    #[test]
    fn test_rectangle_boundary_case() {
        let dx = 2.;
        let dy = 1.;
        let rectangle = NurbsCurve2D::<f64>::polyline(&vec![
            Point2::new(0., 0.),
            Point2::new(dx, 0.),
            Point2::new(dx, dy),
            Point2::new(0., dy),
            Point2::new(0., 0.),
        ]);
        assert!(rectangle.contains(&Point2::new(0., 0.), None).unwrap());
        assert!(rectangle.contains(&Point2::new(dx, 0.), None).unwrap());
        assert!(rectangle.contains(&Point2::new(dx, dy), None).unwrap());
        assert!(rectangle.contains(&Point2::new(0., dy), None).unwrap());

        assert!(!rectangle.contains(&Point2::new(-dx, 0.), None).unwrap());
        assert!(!rectangle.contains(&Point2::new(-dx, dy), None).unwrap());
    }
}