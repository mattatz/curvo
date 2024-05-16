use argmin::core::{CostFunction, Gradient, Hessian};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Matrix2, OVector, U1,
    U2,
};

use crate::{curve::nurbs_curve::NurbsCurve, misc::FloatingPoint};

pub struct CurveIntersectionProblem<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    a: &'a NurbsCurve<T, D>,
    b: &'a NurbsCurve<T, D>,
}

impl<'a, T: FloatingPoint, D: DimName> CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    pub fn new(a: &'a NurbsCurve<T, D>, b: &'a NurbsCurve<T, D>) -> Self {
        CurveIntersectionProblem { a, b }
    }
}

impl<'a, T: FloatingPoint, D: DimName> CostFunction for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = OVector<T, U2>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let da = self.a.knots_domain();
        let db = self.b.knots_domain();
        let pa = param[0].max(da.0).min(da.1);
        let pb = param[1].max(db.0).min(db.1);
        let p0 = self.a.point_at(pa);
        let p1 = self.b.point_at(pb);
        // Ok(OVector::<T, U1>::new((p0 - p1).norm_squared()))
        Ok((p0 - p1).norm_squared())
    }
}

impl<'a, T: FloatingPoint, D: DimName> Gradient for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = OVector<T, U2>;
    type Gradient = OVector<T, U2>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let aderiv = self.a.rational_derivatives(param[0], 1);
        let bderiv = self.b.rational_derivatives(param[1], 1);
        let r = &aderiv[0] - &bderiv[0];
        Ok(
            OVector::<T, U2>::new(aderiv[1].dot(&r), -bderiv[1].dot(&r))
                * T::from_f64(2.0).unwrap(),
        )
    }
}

impl<'a, T: FloatingPoint, D: DimName> Hessian for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = OVector<T, U2>;
    type Hessian = Matrix2<T>;

    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, anyhow::Error> {
        todo!()
    }
}

/*
impl<'a, T: FloatingPoint, D: DimName> Operator for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = DVector<T>;
    type Output = DVector<T>;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let p0 = self.a.point_at(param[0]);
        let p1 = self.b.point_at(param[1]);
        let dist = (p0 - p1).norm_squared();
        Ok(DVector::<T>::from_vec(vec![dist]))
    }
}

impl<'a, T: FloatingPoint, D: DimName> Jacobian for CurveIntersectionProblem<'a, T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    type Param = DVector<T>;
    type Jacobian = DMatrix<T>;
    // type Param = OVector<T, U2>;
    // type Jacobian = Matrix2<T>;

    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, anyhow::Error> {
        // let grad = self.gradient(param)?;
        todo!()
    }
}
*/

#[cfg(test)]
mod tests {

    use nalgebra::{Matrix2, Vector2};

    #[test]
    fn inv() {
        let v2 = Vector2::new(1., 0.);
        let m2 = Matrix2::new(1., 0., 1., 0.);
        let dot = m2 * v2;
        dbg!(dot);
        let d = m2.dot(&m2);
        dbg!(d);
    }
}
