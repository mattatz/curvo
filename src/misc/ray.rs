use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, OVector};

use crate::misc::FloatingPoint;

/// Represents a ray in D dimensions.
#[derive(Clone, Debug)]
pub struct Ray<T: FloatingPoint, D>
where
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    pub(crate) origin: OPoint<T, D>,
    pub(crate) direction: OVector<T, D>,
}

/// Represents the intersection of two rays in D dimensions.
pub struct RayIntersection<T: FloatingPoint, D>
where
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    #[allow(unused)]
    pub(crate) intersection0: (OPoint<T, D>, T),
    #[allow(unused)]
    pub(crate) intersection1: (OPoint<T, D>, T),
}

impl<T: FloatingPoint, D> Ray<T, D>
where
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(origin: OPoint<T, D>, direction: OVector<T, D>) -> Self {
        Self { origin, direction }
    }

    pub fn origin(&self) -> &OPoint<T, D> {
        &self.origin
    }

    pub fn direction(&self) -> &OVector<T, D> {
        &self.direction
    }

    pub fn point_at(&self, t: T) -> OPoint<T, D> {
        &self.origin + &self.direction * t
    }

    /// Finds the intersection between two rays.
    pub fn find_intersection(&self, other: &Self) -> Option<RayIntersection<T, D>> {
        let dab = self.direction.dot(other.direction());
        let daa = self.direction.dot(&self.direction);
        let dbb = other.direction().dot(&other.direction);
        let div = daa * dbb - dab * dab;

        // The rays are parallel.
        if div.abs() < T::default_epsilon() {
            return None;
        }

        let dab0 = self.direction.dot(&other.origin().coords);
        let daa0 = self.direction.dot(&self.origin().coords);
        let dbb0 = other.direction().dot(&other.origin().coords);
        let dba0 = other.direction().dot(&self.origin().coords);

        let num = dab * (dab0 - daa0) - daa * (dbb0 - dba0);
        let w = num / div;
        let t = (dab0 - daa0 + w * dab) / daa;

        Some(RayIntersection {
            intersection0: (self.point_at(t), t),
            intersection1: (other.point_at(w), w),
        })
    }
}
