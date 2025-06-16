pub mod offset_nurbs_curve;
pub use offset_nurbs_curve::*;

/// Corner type for offsetting NURBS curves
#[derive(Debug, Clone, PartialEq)]
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

pub trait Offset<'a, T> {
    type Output;
    type Option;

    fn offset(&'a self, option: Self::Option) -> Self::Output;
}
