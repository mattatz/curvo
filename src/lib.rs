#![allow(clippy::needless_range_loop)]

mod adaptive_tessellation_node;
mod adaptive_tessellation_processor;
mod binomial;
mod bounding_box;
mod bounding_box_traversal;
mod bounding_box_tree;
mod closest_parameter_newton;
mod closest_parameter_problem;
mod curve_length_parameter;
mod floating_point;
mod frenet_frame;
mod invertible;
mod knot_multiplicity;
mod knot_vector;
mod nurbs_curve;
mod nurbs_surface;
mod ray;
mod surface_point;
mod surface_tessellation;
mod transformable;
mod trigonometry;
use closest_parameter_newton::*;
use closest_parameter_problem::*;
use floating_point::*;
use ray::*;
use surface_point::*;

pub mod prelude {
    pub use crate::adaptive_tessellation_processor::AdaptiveTessellationOptions;
    pub use crate::bounding_box::*;
    pub use crate::bounding_box_traversal::*;
    pub use crate::bounding_box_tree::*;
    pub use crate::curve_length_parameter::*;
    pub use crate::floating_point::*;
    pub use crate::frenet_frame::*;
    pub use crate::invertible::*;
    pub use crate::knot_multiplicity::*;
    pub use crate::knot_vector::*;
    pub use crate::nurbs_curve::*;
    pub use crate::nurbs_surface::*;
    pub use crate::surface_tessellation::*;
    pub use crate::transformable::*;
}
