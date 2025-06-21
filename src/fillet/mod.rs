pub mod curve_fillet_option;
pub mod fillet_compound_curve;
pub mod fillet_nurbs_curve;
pub use curve_fillet_option::*;

mod helper;
mod segment;

/// Trait for filleting a geometry
/// O is the option type for the fillet
pub trait Fillet<O> {
    type Output;
    fn fillet(&self, option: O) -> Self::Output;
}
