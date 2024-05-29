use argmin::core::{CostFunction, Gradient};

use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Vector2, U1,
};

use crate::{curve::nurbs_curve::NurbsCurve, misc::FloatingPoint};

pub struct CurveIntersectionProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    a: &'a NurbsCurve<T, D>,
    b: &'a NurbsCurve<T, D>,
}

impl<'a, T: FloatingPoint, D: DimName> CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    pub fn new(a: &'a NurbsCurve<T, D>, b: &'a NurbsCurve<T, D>) -> Self {
        CurveIntersectionProblem { a, b }
    }
}

impl<'a, T: FloatingPoint, D: DimName> Gradient for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = Vector2<T>;
    type Gradient = Vector2<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let du = self.a.rational_derivatives(param[0], 1);
        let dv = self.b.rational_derivatives(param[1], 1);
        let r = &du[0] - &dv[0];
        Ok(Vector2::new(r.dot(&du[1]), -r.dot(&dv[1])) * T::from_f64(2.).unwrap())
    }
}

impl<'a, T: FloatingPoint, D: DimName> CostFunction for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = Vector2<T>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let da = self.a.knots_domain();
        let db = self.b.knots_domain();
        if param[0] < da.0 || da.1 < param[0] || param[1] < db.0 || db.1 < param[1] {
            Err(anyhow::anyhow!("Parameter out of domain"))
        } else {
            let p1 = self.a.point_at(param[0]);
            let p2 = self.b.point_at(param[1]);
            let d = p1 - p2;
            Ok(d.dot(&d))
        }
    }
}
