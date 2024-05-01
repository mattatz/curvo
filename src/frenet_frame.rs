use nalgebra::{IsometryMatrix3, Point3, Rotation3, Translation3, Vector3};

use crate::FloatingPoint;

/// A Frenet frame at a point on a curve.
#[derive(Debug, Clone)]
pub struct FrenetFrame<T: FloatingPoint> {
    position: Point3<T>,
    tangent: Vector3<T>,
    normal: Vector3<T>,
    binormal: Vector3<T>,
}

impl<T: FloatingPoint> FrenetFrame<T> {
    pub fn new(
        position: Point3<T>,
        tangent: Vector3<T>,
        normal: Vector3<T>,
        binormal: Vector3<T>,
    ) -> Self {
        Self {
            position,
            tangent,
            normal,
            binormal,
        }
    }

    pub fn position(&self) -> &Point3<T> {
        &self.position
    }

    pub fn tangent(&self) -> &Vector3<T> {
        &self.tangent
    }

    pub fn normal(&self) -> &Vector3<T> {
        &self.normal
    }

    pub fn binormal(&self) -> &Vector3<T> {
        &self.binormal
    }

    /// Returns the transformation matrix that transforms a target to the frame coordinates.
    pub fn matrix(&self) -> IsometryMatrix3<T> {
        let rot = Rotation3::face_towards(&self.tangent, &self.normal);
        let trans = Translation3::from(self.position);
        trans * rot
    }
}
