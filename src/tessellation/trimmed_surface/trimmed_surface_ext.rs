use nalgebra::{Point3, Vector3, U3, U4};

use crate::{
    misc::FloatingPoint,
    prelude::{
        AdaptiveTessellationNode, AdaptiveTessellationOptions, DividableDirection,
        NurbsSurface3D, SurfaceTessellation3D, Tessellation,
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

/// Implementation for untrimmed NURBS surfaces.
/// Exterior is None (no trim boundary) and interiors are empty,
/// so the entire parametric domain is meshed.
impl<T: FloatingPoint, F> TrimmedSurfaceExt<T, F> for NurbsSurface3D<T>
where
    F: Fn(&AdaptiveTessellationNode<T, U4>) -> Option<DividableDirection> + Copy,
{
    fn knots_domain(&self) -> ((T, T), (T, T)) {
        self.knots_domain()
    }

    fn point_at(&self, u: T, v: T) -> Point3<T> {
        NurbsSurface3D::point_at(self, u, v)
    }

    fn normal_at(&self, u: T, v: T) -> Vector3<T> {
        NurbsSurface3D::normal_at(self, u, v)
    }

    fn exterior(&self) -> Option<&CompoundCurve<T, U3>> {
        None
    }

    fn interiors(&self) -> &[CompoundCurve<T, U3>] {
        &[]
    }

    fn tessellate_base_surface(
        &self,
        options: Option<AdaptiveTessellationOptions<T, U4, F>>,
    ) -> anyhow::Result<SurfaceTessellation3D<T>> {
        Ok(self.tessellate(options))
    }
}
