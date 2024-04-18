use nalgebra::RealField;
use num_traits::ToPrimitive;

/// Trait for floating point types (f32, f64)
pub trait Float: RealField + ToPrimitive + Copy {}

impl Float for f32 {}
impl Float for f64 {}
