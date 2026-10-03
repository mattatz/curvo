use super::{Intersection, IntersectionBFGS};

pub mod compound_curve_intersection;
pub mod curve_intersection_problem;
pub mod curve_intersection_solver_options;
pub mod intersection_compound_curve;
pub mod intersection_curve_curve;

pub use compound_curve_intersection::*;
pub use curve_intersection_problem::*;
pub use curve_intersection_solver_options::*;

/// A struct representing the intersection of two curves.
pub type CurveCurveIntersection<P, T> = Intersection<P, T, T>;

/// The solver for the intersections between NURBS curves, and between a curve and a plane.
pub type CurveIntersectionBFGS<F> = IntersectionBFGS<F>;
