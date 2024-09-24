use nalgebra::{Const, OMatrix, U3};
pub mod compound_curve;
mod curve_direction;
pub use compound_curve::*;

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Invertible, Transformable},
};

/// A closed region bounded by a single exterior curve and zero or more interior curves.
#[derive(Clone, Debug, PartialEq)]
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

    pub fn into_tuple(self) -> (CompoundCurve<T, Const<3>>, Vec<CompoundCurve<T, Const<3>>>) {
        (self.exterior, self.interiors)
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

    pub fn into_interiors(self) -> Vec<CompoundCurve<T, Const<3>>> {
        self.interiors
    }

    pub fn interiors_mut(&mut self) -> &mut Vec<CompoundCurve<T, Const<3>>> {
        &mut self.interiors
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

impl<T: FloatingPoint> From<NurbsCurve<T, U3>> for Region<T> {
    fn from(value: NurbsCurve<T, U3>) -> Self {
        Self::new(value.into(), vec![])
    }
}

impl<T: FloatingPoint> From<CompoundCurve<T, U3>> for Region<T> {
    fn from(value: CompoundCurve<T, U3>) -> Self {
        Self::new(value, vec![])
    }
}
