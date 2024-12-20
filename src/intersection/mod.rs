pub mod compound_curve_intersection;
pub mod curve_intersection;
pub mod curve_intersection_bfgs;
pub mod curve_intersection_problem;
pub mod curve_intersection_solver_options;
pub mod has_intersection;
pub mod intersection_compound_curve;
pub mod intersection_curve;
// pub mod intersection_surface_curve;
// pub mod surface_curve_intersection;

pub use compound_curve_intersection::*;
pub use curve_intersection::*;
pub use curve_intersection_bfgs::*;
pub use curve_intersection_problem::*;
pub use curve_intersection_solver_options::*;
pub use has_intersection::*;
// pub use intersection_surface_curve::*;
// pub use surface_curve_intersection::*;

/// Intersection between two curves trait
pub trait Intersection<'a, T> {
    type Output;
    type Option;

    fn find_intersections(&'a self, other: T, option: Self::Option) -> Self::Output;
}
