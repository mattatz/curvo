use crate::misc::Plane;
use crate::prelude::*;
use argmin::core::{CostFunction, Gradient};
use nalgebra::{Const, OPoint, Vector1};

pub struct CurvePlaneIntersectionProblem<'a, T>
where
    T: FloatingPoint,
{
    curve: &'a NurbsCurve<T, Const<4>>,
    plane: &'a Plane<T>,
}

impl<'a, T> CurvePlaneIntersectionProblem<'a, T>
where
    T: FloatingPoint,
{
    pub fn new(curve: &'a NurbsCurve<T, Const<4>>, plane: &'a Plane<T>) -> Self {
        Self { curve, plane }
    }
}

impl<T: FloatingPoint> CostFunction for CurvePlaneIntersectionProblem<'_, T> {
    type Param = Vector1<T>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let point = self.curve.point_at(param[0]);
        let distance = self.plane.signed_distance(&point);
        Ok(distance * distance) // Return squared distance for better numerical properties
    }
}

impl<T: FloatingPoint> Gradient for CurvePlaneIntersectionProblem<'_, T> {
    type Param = Vector1<T>;
    type Gradient = Vector1<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let t = param[0];
        let deriv = self.curve.rational_derivatives(t, 1);
        let p = deriv[0];
        let v = deriv[1];
        let p_point = OPoint::<T, Const<3>>::from_slice(p.as_slice());
        let distance = self.plane.signed_distance(&p_point);
        let grad = T::from_f64(2.0).unwrap() * distance * self.plane.normal().dot(&v);
        Ok(Vector1::new(grad))
    }
}
