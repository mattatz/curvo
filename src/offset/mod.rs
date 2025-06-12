pub mod offset_nurbs_curve;
pub use offset_nurbs_curve::*;

pub trait Offset<'a, T> {
    type Output;
    type Option;

    fn offset(&'a self, distance: T, option: Self::Option) -> Self::Output;
}
