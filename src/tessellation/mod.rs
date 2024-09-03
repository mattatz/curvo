pub mod adaptive_tessellation_node;
pub mod adaptive_tessellation_option;
pub mod adaptive_tessellation_processor;
pub mod compound_curve;
pub mod curve;
pub mod region;
pub mod surface;
pub mod surface_point;
pub mod surface_tessellation;

pub use surface_point::*;

pub trait Tessellation {
    type Option;
    type Output;
    fn tessellate(&self, options: Self::Option) -> Self::Output;
}
