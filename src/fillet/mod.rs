pub mod fillet_nurbs_curve;
mod segment;
pub use fillet_nurbs_curve::*;

/// Trait for filleting a geometry
/// O is the option type for the fillet
pub trait Fillet<O> {
    type Output;
    fn fillet(&self, option: O) -> Self::Output;
}
