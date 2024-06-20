use argmin::core::{CostFunction, Gradient};

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Vector2, U1,
};

use crate::{curve::nurbs_curve::NurbsCurve, misc::FloatingPoint};

// Gradient & CostFunction provider for finding the intersection between two curves.
pub struct CurveIntersectionProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// The first curve to find the intersection with.
    a: &'a NurbsCurve<T, D>,
    /// The second curve to find the intersection with.
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
        let p1 = self.a.point(param[0]);
        let p2 = self.b.point(param[1]);
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

        /*
        let p1 = self.a.point_at(param[0]);
        let p2 = self.b.point_at(param[1]);
        let d = p1 - p2;
        Ok(d.norm_squared())
        */

        /*
        if param[0] < da.0 || da.1 < param[0] || param[1] < db.0 || db.1 < param[1] {
            Err(anyhow::anyhow!("Parameter out of domain"))
        } else {
            let p1 = self.a.point_at(param[0]);
            let p2 = self.b.point_at(param[1]);
            let d = p1 - p2;
            Ok(d.norm_squared())
        }
        */
    }
}
