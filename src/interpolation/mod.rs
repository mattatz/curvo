use crate::curve::KnotStyle;

pub mod curve;
pub use curve::*;

/// Interpolation trait
pub trait Interpolation {
    type Input;
    type Output;
    fn interpolate(input: &Self::Input, degree: usize) -> Self::Output;
}

/// Periodic interpolation trait
pub trait PeriodicInterpolation {
    type Input;
    type Output;
    fn interpolate_periodic(
        input: &Self::Input,
        degree: usize,
        knot_style: KnotStyle,
    ) -> Self::Output;
}
