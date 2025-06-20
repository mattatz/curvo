pub mod curve_fillet_option;
pub mod fillet_compound_curve;
pub mod fillet_nurbs_curve;
pub use curve_fillet_option::*;
pub use fillet_compound_curve::*;
pub use fillet_nurbs_curve::*;

mod segment;

use crate::misc::FloatingPoint;

/// Trait for filleting a geometry
/// O is the option type for the fillet
pub trait Fillet<O> {
    type Output;
    fn fillet(&self, option: O) -> Self::Output;
}
