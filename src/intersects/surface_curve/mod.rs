pub mod intersection_surface_curve;
pub mod surface_curve_intersection_bfgs;
pub mod surface_curve_intersection_problem;

use nalgebra::Vector3;
pub use surface_curve_intersection_bfgs::*;
pub use surface_curve_intersection_problem::*;

pub type SurfaceCurveParam<T> = Vector3<T>;
pub type SurfaceCurveGradient<T> = Vector3<T>;
