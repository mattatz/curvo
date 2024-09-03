use nalgebra::Const;
pub mod compound_curve;
mod curve_direction;
pub use compound_curve::*;

use crate::misc::FloatingPoint;

/// A closed region bounded by a single exterior curve and zero or more interior curves.
#[derive(Clone, Debug)]
pub struct Region<T: FloatingPoint> {
    exterior: CompoundCurve<T, Const<3>>,
    interiors: Vec<CompoundCurve<T, Const<3>>>,
}

impl<T: FloatingPoint> Region<T> {
    pub fn new(
        exterior: CompoundCurve<T, Const<3>>,
        interiors: Vec<CompoundCurve<T, Const<3>>>,
    ) -> Self {
        Self {
            exterior,
            interiors,
        }
    }

    pub fn exterior(&self) -> &CompoundCurve<T, Const<3>> {
        &self.exterior
    }

    pub fn interiors(&self) -> &[CompoundCurve<T, Const<3>>] {
        &self.interiors
    }
}
