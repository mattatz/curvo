pub mod curve_curve;
pub mod curve_plane;
pub mod has_intersection;
pub mod intersection;
pub mod mesh_plane;
pub mod surface_curve;
pub mod surface_plane;
pub mod surface_tessellation_plane;

pub use curve_curve::*;
pub use curve_plane::*;
pub use has_intersection::*;
pub use intersection::*;
pub use mesh_plane::*;
pub use surface_curve::*;
pub use surface_plane::*;

/// Intersection between two objects trait
pub trait Intersects<'a, T> {
    type Output;
    type Option;

    fn find_intersection(&'a self, other: T, option: Self::Option) -> Self::Output;
}
