use nalgebra::{IsometryMatrix3, Point3, Rotation3, Translation3, Vector3};
use simba::scalar::SupersetOf;

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

    /// Returns the rotation matrix that transforms a target to the frame coordinates.
    pub fn rotation(&self) -> Rotation3<T> {
        Rotation3::face_towards(&self.tangent, &self.normal)
    }

    /// Returns the transformation matrix that transforms a target to the frame coordinates.
    pub fn matrix(&self) -> IsometryMatrix3<T> {
        let trans = Translation3::from(self.position);
        let rot = self.rotation();
        trans * rot
    }

    /// Casts the Frenet frame to another floating point type.
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> FrenetFrame<F> {
        FrenetFrame {
            position: self.position.cast(),
            tangent: self.tangent.cast(),
            normal: self.normal.cast(),
            binormal: self.binormal.cast(),
        }
    }
}
