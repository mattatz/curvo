use nalgebra::{Point3, Vector3, U3, U4};

use crate::{
    misc::FloatingPoint,
    prelude::{
        AdaptiveTessellationNode, AdaptiveTessellationOptions, DividableDirection,
        SurfaceTessellation3D, Tessellation,
    },
    region::CompoundCurve,
    surface::TrimmedSurface,
};

/// A trait for trimmed surface extensions
pub trait TrimmedSurfaceExt<T: FloatingPoint, F> {
    fn knots_domain(&self) -> ((T, T), (T, T));
    fn point_at(&self, u: T, v: T) -> Point3<T>;
    fn normal_at(&self, u: T, v: T) -> Vector3<T>;
    fn exterior(&self) -> Option<&CompoundCurve<T, U3>>;
    fn interiors(&self) -> &[CompoundCurve<T, U3>];

    /// Tessellate the base surface of the trimmed surface
    fn tessellate_base_surface(
        &self,
        options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> anyhow::Result<SurfaceTessellation3D<T>>;
}

/// Implementation of the `TrimmedSurfaceExt` trait for the `TrimmedSurface` type
impl<T: FloatingPoint, F> TrimmedSurfaceExt<T, F> for TrimmedSurface<T>
where
    F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
{
    fn knots_domain(&self) -> ((T, T), (T, T)) {
        self.surface().knots_domain()
    }

    fn point_at(&self, u: T, v: T) -> Point3<T> {
        self.surface().point_at(u, v)
    }

    fn normal_at(&self, u: T, v: T) -> Vector3<T> {
        self.surface().normal_at(u, v)
    }

    fn exterior(&self) -> Option<&CompoundCurve<T, U3>> {
        TrimmedSurface::exterior(self)
    }

    fn interiors(&self) -> &[CompoundCurve<T, U3>] {
        TrimmedSurface::interiors(self)
    }

    fn tessellate_base_surface(
        &self,
        options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> anyhow::Result<SurfaceTessellation3D<T>> {
        Ok(self.surface().tessellate(options))
    }
}
