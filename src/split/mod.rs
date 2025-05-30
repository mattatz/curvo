

pub mod split_nurbs_curve;
pub mod split_nurbs_surface;

pub use split_nurbs_surface::*;

/// Split the object into two objects with the given option
pub trait Split
where
    Self: Sized,
{
    type Option;
    fn try_split(&self, option: Self::Option) -> anyhow::Result<(Self, Self)>;
}
