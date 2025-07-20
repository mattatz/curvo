pub mod decompose_nurbs_curve;
pub mod decompose_nurbs_surface;

/// Decompose a curve or surface into a set of curves or surfaces
pub trait Decompose {
    type Output;
    fn try_decompose(&self) -> anyhow::Result<Self::Output>;
}
