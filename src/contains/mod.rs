use argmin::core::ArgminFloat;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimName, OPoint, Point2, Vector2};

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

        let bb: BoundingBox<T, Const<2>> = self.into();
        let size = bb.size();
        let dx = Vector2::x();
        let sx = size.dot(&dx);

        let ray = NurbsCurve::polyline(&vec![
            point.clone(),
            point + dx * sx * T::from_f64(2.).unwrap(),
        ]);
        self.find_intersections(&ray, option)
            .map(|intersections| intersections.len() % 2 == 1)
    }
}
