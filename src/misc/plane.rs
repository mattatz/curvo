use nalgebra::{convert, Point3, Vector3};
use simba::scalar::SupersetOf;

use crate::misc::FloatingPoint;

/// A plane in 3D space.
#[derive(Debug, Clone, PartialEq)]
pub struct Plane<T: FloatingPoint> {
    normal: Vector3<T>,
    constant: T,
}

impl<T: FloatingPoint> Plane<T> {
    pub fn new(normal: Vector3<T>, constant: T) -> Self {
        Self { normal, constant }
    }

    pub fn normal(&self) -> Vector3<T> {
        self.normal
    }

    pub fn constant(&self) -> T {
        self.constant
    }

    /// Calculate the signed distance from a point to the plane.
    pub fn signed_distance(&self, point: &Point3<T>) -> T {
        self.normal.dot(&point.coords) + self.constant
    }

    /// Cast the plane to a different floating point type.
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> Plane<F> {
        Plane::new(self.normal.cast(), convert(self.constant))
    }
}
