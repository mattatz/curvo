pub mod bounding_box_traversal;
pub mod bounding_box_tree;
pub mod curve_bounding_box_tree;
pub mod surface_bounding_box_tree;

pub use bounding_box_traversal::*;
pub use bounding_box_tree::*;
pub use curve_bounding_box_tree::*;
pub use surface_bounding_box_tree::*;

use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, U1,
};
use simba::scalar::SupersetOf;

use crate::{
    curve::nurbs_curve::NurbsCurve, misc::FloatingPoint, region::CompoundCurve,
    surface::NurbsSurface,
};

/// A struct representing a bounding box in D space.
#[derive(Clone, Debug)]
pub struct BoundingBox<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    min: OVector<T, D>,
    max: OVector<T, D>,
}

impl<T: FloatingPoint, D: DimName> BoundingBox<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a new bounding box from a minimum and maximum point.
    pub fn new(min: OVector<T, D>, max: OVector<T, D>) -> Self {
        let mut tmin = OVector::<T, D>::from_element(T::max_value().unwrap());
        let mut tmax = -tmin.clone();

        for i in 0..D::dim() {
            tmin[i] = tmin[i].min(min[i]).min(max[i]);
            tmax[i] = tmax[i].max(max[i]).max(min[i]);
        }

        BoundingBox {
            min: tmin,
            max: tmax,
        }
    }

    /// Create a new bounding box from point iterator.
    pub fn new_with_points<I: IntoIterator<Item = OPoint<T, D>>>(iter: I) -> Self {
        let mut min = OVector::<T, D>::from_element(T::max_value().unwrap());
        let mut max = -min.clone();

        for point in iter {
            for i in 0..D::dim() {
                min[i] = min[i].min(point[i]);
                max[i] = max[i].max(point[i]);
            }
        }

        Self { min, max }
    }

    pub fn min(&self) -> &OVector<T, D> {
        &self.min
    }

    pub fn max(&self) -> &OVector<T, D> {
        &self.max
    }

    pub fn center(&self) -> OVector<T, D> {
        (&self.min + &self.max) / T::from_usize(2).unwrap()
    }

    pub fn size(&self) -> OVector<T, D> {
        &self.max - &self.min
    }

    /// Check if the bounding box intersects with another bounding box.
    ///
    /// # Examples
    /// ```
    /// use nalgebra::Vector3;
    /// use curvo::prelude::BoundingBox;
    ///
    /// let b0 = BoundingBox::new(Vector3::from_element(0.), Vector3::from_element(1.));
    /// assert!(b0.intersects(&b0, None));
    ///
    /// let eps = 1e-6;
    /// let b1 = BoundingBox::new(Vector3::from_element(0.5), Vector3::from_element(1.5));
    /// assert!(b0.intersects(&b1, None));
    ///
    /// let b2 = BoundingBox::new(Vector3::from_element(1. + eps), Vector3::from_element(2. + eps));
    /// assert!(!b0.intersects(&b2, None));
    /// ```
    pub fn intersects(&self, other: &Self, tolerance: Option<T>) -> bool {
        let tolerance = tolerance.unwrap_or(T::default_epsilon());
        // Check if the bounding boxes intersect along each dimension.
        for i in 0..D::dim() {
            let min0 = self.min[i];
            let max0 = self.max[i];
            let min1 = other.min[i];
            let max1 = other.max[i];

            let a0 = min0 - tolerance;
            let a1 = max0 + tolerance;
            let b0 = min1 - tolerance;
            let b1 = max1 + tolerance;

            let d0 = b0 - a1;
            let d1 = b1 - a0;

            // If the intervals are disjoint,
            // there is no intersection.
            if d0 * d1 > T::zero() {
                return false;
            }
        }

        true
    }

    /// Check if the bounding box contains a point.
    /// # Examples
    /// ```
    /// use nalgebra::{Point3, Vector3};
    /// use curvo::prelude::BoundingBox;
    /// let bb = BoundingBox::new(Vector3::from_element(0.), Vector3::from_element(1.));
    /// assert!(bb.contains(&Point3::new(0.5, 0.5, 0.5)));
    /// assert!(bb.contains(&Point3::new(0., 0.5, 1.0)));
    /// assert!(!bb.contains(&Point3::new(-1e-8, 0.5, 0.5)));
    /// ```
    pub fn contains(&self, point: &OPoint<T, D>) -> bool {
        (0..D::dim()).all(|i| self.min[i] <= point[i] && point[i] <= self.max[i])
    }

    /// Cast the bounding box to a curve with another floating point type
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> BoundingBox<F, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        BoundingBox {
            min: self.min.clone().cast(),
            max: self.max.clone().cast(),
        }
    }

    /// Get corner points of the bounding box.
    pub fn corners(&self) -> Vec<OPoint<T, D>> {
        let dim = D::dim() as u32;
        let mut corners = Vec::with_capacity(2usize.pow(dim));
        for i in 0..2usize.pow(dim) {
            let mut point = OPoint::<T, D>::origin();
            for j in 0..D::dim() {
                point[j] = if i & (1 << j) == 0 {
                    self.min[j]
                } else {
                    self.max[j]
                };
            }
            corners.push(point);
        }
        corners
    }

    /// Get lines of the bounding box.
    pub fn lines(&self) -> Vec<(OPoint<T, D>, OPoint<T, D>)> {
        let corners = self.corners();
        let mut lines = Vec::with_capacity(2usize.pow(D::dim() as u32));
        let pow = 2usize.pow(D::dim() as u32);
        for i in 0..D::dim() {
            for j in 0..pow {
                let a = corners[j].clone();
                let b = corners[j ^ (1 << i)].clone();
                lines.push((a, b));
            }
        }
        lines
    }
}

impl<T: FloatingPoint, D: DimName> FromIterator<OPoint<T, D>> for BoundingBox<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = OPoint<T, D>>>(iter: I) -> Self {
        Self::new_with_points(iter)
    }
}

impl<'a, T: FloatingPoint, D: DimName> From<&'a NurbsCurve<T, D>>
    for BoundingBox<T, DimNameDiff<D, U1>>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: &'a NurbsCurve<T, D>) -> Self {
        let pts = value.dehomogenized_control_points();
        Self::new_with_points(pts)
    }
}

impl<'a, T: FloatingPoint, D: DimName> From<&'a CompoundCurve<T, D>>
    for BoundingBox<T, DimNameDiff<D, U1>>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: &'a CompoundCurve<T, D>) -> Self {
        let pts = value
            .spans()
            .iter()
            .flat_map(|span| span.dehomogenized_control_points());
        Self::from_iter(pts)
    }
}

impl<'a, T: FloatingPoint, D: DimName> From<&'a NurbsSurface<T, D>>
    for BoundingBox<T, DimNameDiff<D, U1>>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: &'a NurbsSurface<T, D>) -> Self {
        let pts = value.dehomogenized_control_points();
        let flatten = pts.into_iter().flatten();
        Self::new_with_points(flatten)
    }
}
