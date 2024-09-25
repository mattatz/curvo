use argmin::core::{CostFunction, Gradient, Hessian};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector,
    Vector2, U1,
};

use crate::{misc::FloatingPoint, surface::NurbsSurface};

/// Gradient & Hessian provider for finding the closest parameter on a surface to a given point.
pub struct SurfaceClosestParameterProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// The point to find the closest parameter to.
    point: &'a OPoint<T, DimNameDiff<D, U1>>,
    /// The surface to find the closest parameter on.
    surface: &'a NurbsSurface<T, D>,
}

impl<'a, T: FloatingPoint, D: DimName> SurfaceClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new(point: &'a OPoint<T, DimNameDiff<D, U1>>, surface: &'a NurbsSurface<T, D>) -> Self {
        SurfaceClosestParameterProblem { point, surface }
    }
}

impl<'a, T: FloatingPoint, D: DimName> CostFunction for SurfaceClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = Vector2<T>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let p = self.surface.point_at(param.x, param.y);
        let d = p - self.point;
        Ok(d.norm())
    }
}

impl<'a, T: FloatingPoint, D: DimName> Gradient for SurfaceClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = Vector2<T>;
    type Gradient = OVector<T, DimNameDiff<D, U1>>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let p = self.surface.point_at(param.x, param.y);
        let d = p - self.point;
        Ok(d)
    }
}

impl<'a, T: FloatingPoint, D: DimName> Hessian for SurfaceClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = Vector2<T>;
    type Hessian = Vec<Vec<OVector<T, DimNameDiff<D, U1>>>>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, anyhow::Error> {
        Ok(self.surface.rational_derivatives(param.x, param.y, 2))
    }
}
