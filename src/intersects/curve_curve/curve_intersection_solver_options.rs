use crate::misc::FloatingPoint;

/// Hyperparameters for the curve intersection solver.
#[derive(Clone, Debug)]
pub struct CurveIntersectionSolverOptions<T: FloatingPoint> {
    /// Minimum distance between two points to consider them as intersecting.
    pub minimum_distance: T,
    /// Knot domain division for the threshold of the bounding box tree.
    /// Before detecting intersections between curves, perform intersection detection between the bounding boxes that enclose the curves as a sub-problem.
    /// Simultaneously divide the bounding boxes and the curves into smaller segments.
    pub knot_domain_division: usize,
    /// Tolerance for the step size in the line search.
    pub step_size_tolerance: T,
    /// Tolerance for the cost function to determine convergence.
    pub cost_tolerance: T,
    /// Maximum number of iterations for the Newton method.
    pub max_iters: u64,
}

impl<T: FloatingPoint> Default for CurveIntersectionSolverOptions<T> {
    fn default() -> Self {
        Self {
            minimum_distance: T::from_f64(1e-5).unwrap(),
            knot_domain_division: 64,
            step_size_tolerance: T::from_f64(1e-8).unwrap(),
            cost_tolerance: T::from_f64(1e-10).unwrap(),
            max_iters: 200,
        }
    }
}

impl<T: FloatingPoint> CurveIntersectionSolverOptions<T> {
    pub fn with_minimum_distance(mut self, minimum_distance: T) -> Self {
        self.minimum_distance = minimum_distance;
        self
    }

    pub fn with_knot_domain_division(mut self, knot_domain_division: usize) -> Self {
        self.knot_domain_division = knot_domain_division;
        self
    }

    pub fn with_step_size_tolerance(mut self, step_size_tolerance: T) -> Self {
        self.step_size_tolerance = step_size_tolerance;
        self
    }

    pub fn with_cost_tolerance(mut self, cost_tolerance: T) -> Self {
        self.cost_tolerance = cost_tolerance;
        self
    }

    pub fn with_max_iters(mut self, max_iters: u64) -> Self {
        self.max_iters = max_iters;
        self
    }
}
