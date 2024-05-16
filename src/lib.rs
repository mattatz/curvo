#![allow(clippy::needless_range_loop)]

mod bounding_box;
mod closest_parameter;
mod curve;
mod intersection;
mod knot;
mod misc;
mod surface;
mod tessellation;
use closest_parameter::*;

pub mod prelude {
    pub use crate::bounding_box::*;
    pub use crate::curve::*;
    pub use crate::intersection::*;
    pub use crate::knot::*;
    pub use crate::misc::*;
    pub use crate::surface::*;
    pub use crate::tessellation::adaptive_tessellation_processor::AdaptiveTessellationOptions;
    pub use crate::tessellation::surface_tessellation::*;
}
