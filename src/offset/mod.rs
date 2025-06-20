pub mod curve_offset_option;
mod helper;
pub mod offset_compound_curve;
pub mod offset_nurbs_curve;
mod vertex;
pub use curve_offset_option::*;

/// Corner type for offsetting NURBS curves
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurveOffsetCornerType {
    /// Just offset the curve segments
    None,
    /// Extend the two segments & trim at the intersection
    Sharp,
    /// Insert arc at the corner
    Round,
    /// Insert 3-degree bezier curve at the corner
    Smooth,
    /// Insert chamfer at the corner
    Chamfer,
}

impl Default for CurveOffsetCornerType {
    fn default() -> Self {
        Self::Sharp
    }
}

/// Trait for offsetting a geometry
pub trait Offset<'a, T> {
    type Output;
    type Option;

    fn offset(&'a self, option: Self::Option) -> Self::Output;
}
