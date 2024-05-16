use nalgebra::{Point, SVector};

use crate::misc::FloatingPoint;

/// Represents a ray in D dimensions.
pub struct Ray<T: FloatingPoint, const D: usize> {
    pub(crate) origin: Point<T, D>,
    pub(crate) direction: SVector<T, D>,
}

/// Represents the intersection of two rays in D dimensions.
pub struct RayIntersection<T: FloatingPoint, const D: usize> {
    #[allow(unused)]
    pub(crate) intersection0: (Point<T, D>, T),
    #[allow(unused)]
    pub(crate) intersection1: (Point<T, D>, T),
}

impl<T: FloatingPoint, const D: usize> Ray<T, D> {
    pub fn new(origin: Point<T, D>, direction: SVector<T, D>) -> Self {
        Self { origin, direction }
    }

    pub fn origin(&self) -> &Point<T, D> {
        &self.origin
    }

    pub fn direction(&self) -> &SVector<T, D> {
        &self.direction
    }

    pub fn point_at(&self, t: T) -> Point<T, D> {
        self.origin + self.direction * t
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
