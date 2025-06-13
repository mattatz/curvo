pub mod offset_nurbs_curve;
pub use offset_nurbs_curve::*;

#[derive(Debug, Clone, PartialEq)]
pub enum CurveOffsetCornerType {
    None,
    Sharp,
    Round,
    Smooth,
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
