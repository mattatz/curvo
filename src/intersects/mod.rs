pub mod curve_curve;
pub mod has_intersection;
pub mod intersection;
pub mod surface_curve;

pub use curve_curve::*;
pub use has_intersection::*;
pub use intersection::*;

/// Intersection between two objects trait
pub trait Intersects<'a, T> {
    type Output;
    type Option;

    fn find_intersections(&'a self, other: T, option: Self::Option) -> Self::Output;
}
