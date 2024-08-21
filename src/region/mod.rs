use nalgebra::Const;
pub mod compound_curve;
pub use compound_curve::*;

use crate::misc::FloatingPoint;

/// A closed region bounded by a single exterior curve and zero or more interior curves.
#[derive(Clone, Debug)]
pub struct Region<T: FloatingPoint> {
    exterior: CompoundCurve<T, Const<2>>,
    interiors: Vec<CompoundCurve<T, Const<2>>>,
}
