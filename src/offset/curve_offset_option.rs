use crate::{misc::FloatingPoint, offset::CurveOffsetCornerType};

/// Offset option for NURBS curves
#[derive(Debug, Clone, PartialEq)]
pub struct CurveOffsetOption<T> {
    /// Offset distance
    distance: T,
    /// Normal tolerance for tessellation
    normal_tolerance: T,
    /// Knot tolerance for reducing knots
    knot_tolerance: T,
    /// Corner type
    corner_type: CurveOffsetCornerType,
}

impl<T: FloatingPoint> Default for CurveOffsetOption<T> {
    fn default() -> Self {
        Self {
            distance: T::zero(),
            normal_tolerance: T::from_f64(1e-4).unwrap(),
            knot_tolerance: T::from_f64(1e-4).unwrap(),
            corner_type: Default::default(),
        }
    }
}

impl<T> CurveOffsetOption<T> {
    pub fn distance(&self) -> &T {
        &self.distance
    }

    pub fn normal_tolerance(&self) -> &T {
        &self.normal_tolerance
    }

    pub fn knot_tolerance(&self) -> &T {
        &self.knot_tolerance
    }

    pub fn corner_type(&self) -> CurveOffsetCornerType {
        self.corner_type
    }

    pub fn with_distance(mut self, distance: T) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_normal_tolerance(mut self, tol: T) -> Self {
        self.normal_tolerance = tol;
        self
    }

    pub fn with_knot_tolerance(mut self, tol: T) -> Self {
        self.knot_tolerance = tol;
        self
    }

    pub fn with_corner_type(mut self, ty: CurveOffsetCornerType) -> Self {
        self.corner_type = ty;
        self
    }
}
