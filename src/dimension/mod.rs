use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameSum, OPoint, U1,
};

use crate::{curve::NurbsCurve, misc::FloatingPoint, region::CompoundCurve};

/// Elevate the dimension of the geometry
pub trait ElevateDimension {
    type Output;

    fn elevate_dimension(&self) -> Self::Output;
}

impl<T: FloatingPoint, D: DimName> ElevateDimension for NurbsCurve<T, D>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameSum<D, U1>>,
{
    type Output = NurbsCurve<T, DimNameSum<D, U1>>;

    /// Elevate the dimension of the curve (e.g., 2D -> 3D)
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// let points: Vec<Point2<f64>> = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(-1.0, 1.0),
    /// ];
    /// let curve2d = NurbsCurve2D::interpolate(&points, 3).unwrap();
    /// let curve3d: NurbsCurve3D<f64> = curve2d.elevate_dimension();
    /// let (start, end) = curve2d.knots_domain();
    /// let (p0, p1) = (curve2d.point_at(start), curve2d.point_at(end));
    /// let (p2, p3) = (curve3d.point_at(start), curve3d.point_at(end));
    /// assert_eq!(p0.x, p2.x);
    /// assert_eq!(p0.y, p2.y);
    /// assert_eq!(p2.z, 0.0);
    /// assert_eq!(p1.x, p3.x);
    /// assert_eq!(p1.y, p3.y);
    /// assert_eq!(p3.z, 0.0);
    /// ```
    fn elevate_dimension(&self) -> NurbsCurve<T, DimNameSum<D, U1>>
    where
        D: DimNameAdd<U1>,
        DefaultAllocator: Allocator<DimNameSum<D, U1>>,
    {
        let mut control_points = vec![];
        for p in self.control_points().iter() {
            let mut coords = vec![];
            for i in 0..(D::dim() - 1) {
                coords.push(p[i]);
            }
            coords.push(T::zero()); // set a zero in the last dimension
            coords.push(p[D::dim() - 1]);
            control_points.push(OPoint::from_slice(&coords));
        }

        NurbsCurve::new_unchecked(self.degree(), control_points, self.knots().clone())
    }
}

impl<T: FloatingPoint, D: DimName> ElevateDimension for CompoundCurve<T, D>
where
    D: DimNameAdd<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameSum<D, U1>>,
{
    type Output = CompoundCurve<T, DimNameSum<D, U1>>;

    /// Elevate the dimension of the compound curve (e.g., 2D -> 3D)
    fn elevate_dimension(&self) -> Self::Output {
        let spans = self.spans().iter().map(|s| s.elevate_dimension()).collect();
        CompoundCurve::new_unchecked(spans)
    }
}
