pub mod nurbs_surface;
// pub mod trimmed_surface;
pub use nurbs_surface::*;
// pub use trimmed_surface::*;

/// The direction in the UV space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UVDirection {
    U,
    V,
}

/// The direction to flip the surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    U,
    V,
    UV,
}
