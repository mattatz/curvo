pub mod intersection_surface_curve;
pub mod surface_curve_intersection_problem;

use nalgebra::Vector3;
pub use surface_curve_intersection_problem::*;

pub type SurfaceCurveParam<T> = Vector3<T>;
pub type SurfaceCurveGradient<T> = Vector3<T>;

/// The solver for the intersections between a NURBS surface and a curve.
pub type SurfaceCurveIntersectionBFGS<F> = super::IntersectionBFGS<F>;
