pub mod adaptive_tessellation_node;
pub mod adaptive_tessellation_option;
pub mod adaptive_tessellation_processor;
pub mod surface;
pub mod surface_point;
pub mod surface_tessellation;
pub mod tessellation_compound_curve;
pub mod tessellation_curve;
pub mod tessellation_region;
pub mod trimmed_surface;

pub use surface_point::*;

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
    fn constrained_tessalate(
        &self,
        constraints: Self::Constraint,
        options: Self::Option,
    ) -> Self::Output;
}
