use nalgebra::{Const, OMatrix, U3};
pub mod compound_curve;
mod curve_direction;
pub use compound_curve::*;

use crate::misc::{FloatingPoint, Invertible, Transformable};

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

    pub fn into_exterior(self) -> CompoundCurve<T, Const<3>> {
        self.exterior
    }

    pub fn interiors(&self) -> &[CompoundCurve<T, Const<3>>] {
        &self.interiors
    }
}

impl<'a, T: FloatingPoint> Transformable<&'a OMatrix<T, U3, U3>> for Region<T> {
    fn transform(&mut self, transform: &'a OMatrix<T, U3, U3>) {
        self.exterior.transform(transform);
        self.interiors
            .iter_mut()
            .for_each(|interior| interior.transform(transform));
    }
}

impl<T: FloatingPoint> Invertible for Region<T> {
    fn invert(&mut self) {
        self.exterior.invert();
        self.interiors
            .iter_mut()
            .for_each(|interior| interior.invert());
    }
}
