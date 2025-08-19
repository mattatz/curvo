use crate::misc::Plane;
use crate::prelude::*;
use argmin::core::{CostFunction, Gradient};
use nalgebra::{Const, OPoint, Vector2};

pub struct SurfacePlaneIntersectionProblem<'a, T>
where
    T: FloatingPoint,
{
    surface: &'a NurbsSurface<T, Const<4>>,
    plane: &'a Plane<T>,
}

impl<'a, T> SurfacePlaneIntersectionProblem<'a, T>
where
    T: FloatingPoint,
{
    pub fn new(surface: &'a NurbsSurface<T, Const<4>>, plane: &'a Plane<T>) -> Self {
        Self { surface, plane }
    }
}

impl<T: FloatingPoint> CostFunction for SurfacePlaneIntersectionProblem<'_, T> {
    type Param = Vector2<T>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, anyhow::Error> {
        let point = self.surface.point_at(param[0], param[1]);
        let distance = self.plane.signed_distance(&point);
        Ok(distance * distance) // Return squared distance for better numerical properties
    }
}

impl<T: FloatingPoint> Gradient for SurfacePlaneIntersectionProblem<'_, T> {
    type Param = Vector2<T>;
    type Gradient = Vector2<T>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, anyhow::Error> {
        let u = param[0];
        let v = param[1];
        let derivs = self.surface.rational_derivatives(u, v, 1);
        let p = &derivs[0][0];
        let du = &derivs[1][0];
        let dv = &derivs[0][1];
        
        let p_point = OPoint::<T, Const<3>>::from_slice(p.as_slice());
        let distance = self.plane.signed_distance(&p_point);
        let normal = self.plane.normal();
        
        let grad_u = T::from_f64(2.0).unwrap() * distance * normal.dot(du);
        let grad_v = T::from_f64(2.0).unwrap() * distance * normal.dot(dv);
        
        Ok(Vector2::new(grad_u, grad_v))
    }
}