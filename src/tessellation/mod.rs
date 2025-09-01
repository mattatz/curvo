pub mod adaptive_tessellation_node;
pub mod adaptive_tessellation_option;
pub mod adaptive_tessellation_processor;
pub mod boundary_constraints;
mod cardinal_direction;
pub mod surface;
pub mod surface_point;
pub mod surface_tessellation;
pub mod tessellation_compound_curve;
pub mod tessellation_curve;
pub mod tessellation_region;
pub mod trimmed_surface;
pub mod trimmed_surface_constraints;

pub use adaptive_tessellation_node::DividableDirection;
pub use surface_point::*;

use crate::tessellation::adaptive_tessellation_node::AdaptiveTessellationNode;

pub type DefaultDivider<T, D> = fn(&AdaptiveTessellationNode<T, D>) -> Option<DividableDirection>;

/// A trait for tessellating a shape
pub trait Tessellation {
    type Option;
    type Output;
    fn tessellate(&self, options: Self::Option) -> Self::Output;
}

/// A trait for tessellating a shape with constraints
pub trait ConstrainedTessellation {
    type Constraint;
    type Option;
    type Output;
    fn constrained_tessellate(
        &self,
        constraints: Self::Constraint,
        options: Self::Option,
    ) -> Self::Output;
}
