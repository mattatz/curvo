pub mod nurbs_surface;
pub use nurbs_surface::*;

/// The direction to flip the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    U,
    V,
    UV,
}
