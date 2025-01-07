#![allow(clippy::needless_range_loop)]

mod boolean;
mod bounding_box;
mod closest_parameter;
mod contains;
mod curve;
mod intersects;
mod knot;
mod misc;
mod polygon_mesh;
mod region;
mod split;
mod surface;
mod tessellation;
mod trim;
use closest_parameter::*;

pub mod prelude {
    pub use crate::boolean::*;
    pub use crate::bounding_box::*;
    pub use crate::contains::*;
    pub use crate::curve::*;
    pub use crate::intersects::*;
    pub use crate::knot::*;
    pub use crate::misc::*;
    pub use crate::polygon_mesh::*;
    pub use crate::region::*;
    pub use crate::split::*;
    pub use crate::surface::*;
    pub use crate::tessellation::{
        adaptive_tessellation_option::AdaptiveTessellationOptions,
        boundary_constraints::BoundaryConstraints, surface_tessellation::*,
        ConstrainedTessellation, Tessellation,
    };
    pub use crate::trim::*;
}
