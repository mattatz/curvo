use argmin::core::{Gradient, Hessian};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{curve::nurbs_curve::NurbsCurve, misc::FloatingPoint};

/// Gradient & Hessian provider for finding the closest parameter on a curve to a given point.
pub struct ClosestParameterProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// The point to find the closest parameter to.
    point: &'a OPoint<T, DimNameDiff<D, U1>>,
    /// The curve to find the closest parameter on.
    curve: &'a NurbsCurve<T, D>,
}

impl<'a, T: FloatingPoint, D: DimName> ClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub fn new(point: &'a OPoint<T, DimNameDiff<D, U1>>, curve: &'a NurbsCurve<T, D>) -> Self {
        ClosestParameterProblem { point, curve }
    }
}

impl<'a, T: FloatingPoint, D: DimName> Gradient for ClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = T;
    type Gradient = T;

    /// C'(u) * ( C(u) - P )
    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let e = self.curve.rational_derivatives(*param, 1);
        let d = &e[0] - &self.point.coords;
        let f = e[1].dot(&d);
        Ok(f)
    }
}

impl<'a, T: FloatingPoint, D: DimName> Hessian for ClosestParameterProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = T;
    type Hessian = T;

    /// C"(u) * ( C(u) - p ) + C'(u) * C'(u)
    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, anyhow::Error> {
        let e = self.curve.rational_derivatives(*param, 2);
        let d = &e[0] - &self.point.coords;
        let s0 = e[2].dot(&d);
        let s1 = e[1].dot(&e[1]);
        Ok(s0 + s1)
    }
}
