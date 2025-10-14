pub mod curve_morph;
pub mod point_morph;
pub mod surface_morph;

#[cfg(test)]
mod tests;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{misc::FloatingPoint, surface::nurbs_surface::NurbsSurface};

/// A unified trait for morphing points, curves, and surfaces from a reference surface to a target surface.
///
/// This trait enables the transformation of geometric entities by:
/// 1. Finding the closest UV parameter(s) on the reference surface
/// 2. Evaluating the target surface at the same UV parameter(s)
/// 3. Returning the morphed entity
///
/// # Type Parameters
///
/// * `T` - The floating point type (f32 or f64)
/// * `D` - The dimension of the homogeneous coordinate system
///
/// # Implementations
///
/// This trait is implemented for:
/// - `OPoint<T, Const<3>>` (Point3) - Morphs a single point
/// - `NurbsCurve<T, Const<4>>` - Morphs all control points of a curve
/// - `NurbsSurface<T, Const<4>>` - Morphs all control points of a surface
pub trait Morph<T, D>
where
    T: FloatingPoint,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// The output type after morphing
    type Output;

    /// Morphs the geometric entity from the reference surface to the target surface.
    ///
    /// # Arguments
    ///
    /// * `reference_surface` - The reference surface to find UV parameter(s)
    /// * `target_surface` - The target surface to evaluate at the found UV parameter(s)
    ///
    /// # Returns
    ///
    /// The morphed entity, or an error if the morphing fails
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, D>,
        target_surface: &NurbsSurface<T, D>,
    ) -> anyhow::Result<Self::Output>;
}
