use argmin::core::{CostFunction, Gradient};

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Vector3, U1,
};

use crate::{curve::nurbs_curve::NurbsCurve, misc::FloatingPoint, surface::NurbsSurface};

use super::{SurfaceCurveGradient, SurfaceCurveParam};

// Gradient & CostFunction provider for finding the intersection between surface & curve.
pub struct SurfaceCurveIntersectionProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    /// The first surface to find the intersection with.
    a: &'a NurbsSurface<T, D>,
    /// The second curve to find the intersection with.
    b: &'a NurbsCurve<T, D>,
}

impl<'a, T: FloatingPoint, D: DimName> SurfaceCurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(a: &'a NurbsSurface<T, D>, b: &'a NurbsCurve<T, D>) -> Self {
        SurfaceCurveIntersectionProblem { a, b }
    }
}

impl<'a, T: FloatingPoint, D: DimName> Gradient for SurfaceCurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = SurfaceCurveParam<T>;
    type Gradient = SurfaceCurveGradient<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let dc = self.b.rational_derivatives(param.x, 1);
        let ds = self.a.rational_derivatives(param.y, param.z, 1);
        let r = &ds[0][0] - &dc[0];
        let drdu = &ds[1][0];
        let drdv = &ds[0][1];
        let drdt = -&dc[1];
        Ok(Vector3::new(drdt.dot(&r), drdu.dot(&r), drdv.dot(&r)) * T::from_f64(2.).unwrap())
    }
}

impl<'a, T: FloatingPoint, D: DimName> CostFunction for SurfaceCurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Param = SurfaceCurveParam<T>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let p1 = self.a.point(param.y, param.z);
        let p2 = self.b.point(param.x);
        let c1 = p1.coords;
        let c2 = p2.coords;
        let idx = D::dim() - 1;
        let w1 = c1[idx];
        let w2 = c2[idx];

        if w1 != T::zero() && w2 != T::zero() {
            let v1 =
                c1.generic_view((0, 0), (<D as DimNameSub<U1>>::Output::name(), Const::<1>)) / w1;
            let v2 =
                c2.generic_view((0, 0), (<D as DimNameSub<U1>>::Output::name(), Const::<1>)) / w2;
            let d = v1 - v2;
            Ok(d.norm_squared())
        } else {
            Err(anyhow::anyhow!("Parameter out of domain"))
        }
    }
}
