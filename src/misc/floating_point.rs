use nalgebra::RealField;
use num_traits::ToPrimitive;

/// Trait for floating point types (f32, f64)
/// Mainly used to identify the type of the field in nalgebra
pub trait FloatingPoint: RealField + ToPrimitive + Copy {}

impl FloatingPoint for f32 {}
impl FloatingPoint for f64 {}
