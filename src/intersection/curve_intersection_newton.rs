use argmin::{argmin_error_closure, core::*, float};
use argmin_math::ArgminDot;
use nalgebra::{Matrix2, Vector2};

use crate::misc::FloatingPoint;

/// Customized Newton's method for finding the intersections between NURBS curves
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct CurveIntersectionNewton<F> {
    /// Tolerance for the step size in the line search
    step_size_tolerance: F,

    /// Tolerance for the cost function to determine convergence
    cost_tolerance: F,

    /// Maximum number of iterations for line search
    line_search_max_iters: u64,
}

impl<F> Default for CurveIntersectionNewton<F>
where
    F: FloatingPoint,
{
    fn default() -> Self {
        Self {
            step_size_tolerance: float!(1e-8),
            cost_tolerance: float!(1e-8),
            line_search_max_iters: 32,
        }
    }
}

impl<F> CurveIntersectionNewton<F>
where
    F: FloatingPoint,
{
    /// Construct a new instance of [`Newton`]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_step_size_tolerance(mut self, tolerance: F) -> Self {
        self.step_size_tolerance = tolerance;
        self
    }

    pub fn with_cost_tolerance(mut self, tolerance: F) -> Self {
        self.cost_tolerance = tolerance;
        self
    }

    pub fn with_line_search_max_iters(mut self, line_search_max_iters: u64) -> Self {
        self.line_search_max_iters = line_search_max_iters;
        self
    }
}

impl<'a, O, F> Solver<O, IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>>
    for CurveIntersectionNewton<F>
where
    O: Gradient<Param = Vector2<F>, Gradient = Vector2<F>>
        + CostFunction<Param = Vector2<F>, Output = F>,
    F: FloatingPoint + ArgminFloat,
{
    const NAME: &'static str = "Curve intersection newton method with line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> Result<
        (
            IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
            Option<KV>,
        ),
        Error,
    > {
        let x0 = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = problem.cost(x0)?;

        Ok((state.cost(cost), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> Result<
        (
            IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
            Option<KV>,
        ),
        Error,
    > {
        let x0 = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let f0 = state.get_cost();

        let g0 = match state.get_gradient() {
            Some(prev) => *prev,
            None => problem.gradient(x0)?,
        };

        let h0 = state.get_hessian().cloned().unwrap_or(Matrix2::identity());

        // line search
        let step = -h0 * g0;

        let norm = step.norm();
        // println!("norm: {}", norm);

        let mut t = F::one();
        let df0 = g0.dot(&step);
        let mut x1 = *x0;
        let mut f1 = anyhow::Ok(f0);

        let dt = F::from_f64(1e-1).unwrap();
        let dec = F::from_f64(0.5).unwrap();
        for _ in 0..self.line_search_max_iters {
            if t * norm < self.step_size_tolerance {
                return Ok((state.cost(f0), None));
            }

            let s = step * t;
            x1 = x0 + s;
            f1 = problem.cost(&x1);
            if match f1 {
                Ok(f1) => f1 - f0 >= dt * t * df0,
                _ => true,
            } {
                t *= dec;
            } else {
                break;
            }
        }

        // println!("{}: {}, {}", state.iter, x1, f1);

        let f1 = f1?;

        let g1 = problem.gradient(&x1)?;
        let y = g1 - g0;
        let s = step * t;
        let ys = y.dot(&s);
        let s_t = s * s.transpose();
        let hy = h0 * y;

        let h1 = (h0 + s_t * ((ys + y.dot(&hy)) / (ys * ys)))
            - (((hy * s.transpose()) + (s * hy.transpose())) / ys);

        Ok((state.param(x1).cost(f1).gradient(g1).hessian(h1), None))
    }

    fn terminate(
        &mut self,
        state: &IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> TerminationStatus {
        if state.iter > state.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }

        if let Some(g) = state.get_gradient() {
            if g.x.is_nan() || g.y.is_nan() || g.x.is_infinite() || g.y.is_infinite() {
                return TerminationStatus::Terminated(TerminationReason::SolverExit(
                    "gradient is NaN or infinite".into(),
                ));
            }
        }

        if let Some(h) = state.get_hessian() {
            if h[(0, 0)].is_nan()
                || h[(0, 1)].is_nan()
                || h[(1, 0)].is_nan()
                || h[(1, 1)].is_nan()
                || h[(0, 0)].is_infinite()
                || h[(0, 1)].is_infinite()
                || h[(1, 0)].is_infinite()
                || h[(1, 1)].is_infinite()
            {
                return TerminationStatus::Terminated(TerminationReason::SolverExit(
                    "hessian is NaN or infinite".into(),
                ));
            }
        }

        if nalgebra::ComplexField::abs(state.get_cost() - state.get_prev_cost())
            < self.cost_tolerance
            || nalgebra::ComplexField::abs(state.get_best_cost() - state.get_prev_best_cost())
                < self.cost_tolerance
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }

        TerminationStatus::NotTerminated
    }
}
