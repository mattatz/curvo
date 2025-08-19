use argmin::{argmin_error_closure, core::*, float};
use nalgebra::{Matrix2, Vector2};

use crate::misc::FloatingPoint;

/// Customized quasi-Newton's method for finding the intersections between NURBS surface & plane
#[derive(Clone, Copy)]
pub struct SurfacePlaneIntersectionBFGS<F> {
    /// Tolerance for the step size in the line search
    step_size_tolerance: F,

    /// Tolerance for the cost function to determine convergence
    cost_tolerance: F,
}

impl<F> Default for SurfacePlaneIntersectionBFGS<F>
where
    F: FloatingPoint,
{
    fn default() -> Self {
        Self {
            step_size_tolerance: float!(1e-8),
            cost_tolerance: float!(1e-8),
        }
    }
}

impl<F> SurfacePlaneIntersectionBFGS<F>
where
    F: FloatingPoint,
{
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

    pub fn step_size_tolerance(&self) -> F {
        self.step_size_tolerance
    }

    pub fn cost_tolerance(&self) -> F {
        self.cost_tolerance
    }
}

type SurfacePlaneIterState<F> = IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>;

impl<O, F> Solver<O, SurfacePlaneIterState<F>> for SurfacePlaneIntersectionBFGS<F>
where
    O: Gradient<Param = Vector2<F>, Gradient = Vector2<F>>
        + CostFunction<Param = Vector2<F>, Output = F>,
    F: FloatingPoint + ArgminFloat,
{
    const NAME: &'static str = "Surface plane intersection newton method with line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: SurfacePlaneIterState<F>,
    ) -> Result<(SurfacePlaneIterState<F>, Option<KV>), Error> {
        let x0 = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`SurfacePlaneIntersectionBFGS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = problem.cost(x0)?;

        Ok((state.cost(cost), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: SurfacePlaneIterState<F>,
    ) -> Result<(SurfacePlaneIterState<F>, Option<KV>), Error> {
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

        let mut t = F::one();
        let df0 = g0.dot(&step);
        let mut x1 = *x0;
        let mut f1 = anyhow::Ok(f0);

        let dt = F::from_f64(1e-1).unwrap();
        let dec = F::from_f64(0.5).unwrap();
        let mut it = 0;
        let max_iters = state.get_max_iters();
        for _ in 0..max_iters {
            it += 1;
            if t * norm < self.step_size_tolerance() {
                break;
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

        let f1 = f1.unwrap_or(f0);

        let g1 = problem.gradient(&x1)?;
        let y = g1 - g0;
        let s = step * t;
        let ys = y.dot(&s);
        let s_t = s * s.transpose();
        let hy = h0 * y;

        let h1 = (h0 + s_t * ((ys + y.dot(&hy)) / (ys * ys)))
            - (((hy * s.transpose()) + (s * hy.transpose())) / ys);

        Ok((
            state
                .param(x1)
                .cost(f1)
                .gradient(g1)
                .hessian(h1)
                .max_iters(max_iters - it), // decrease remaining iterations by # of line search iterations
            None,
        ))
    }

    fn terminate(&mut self, state: &SurfacePlaneIterState<F>) -> TerminationStatus {
        if state.iter > state.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }

        if let Some(g) = state.get_gradient() {
            if g.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return TerminationStatus::Terminated(TerminationReason::SolverExit(
                    "gradient is NaN or infinite".into(),
                ));
            }
        }

        if let Some(h) = state.get_hessian() {
            let has_nan = h.iter().any(|&v| v.is_nan() || v.is_infinite());
            if has_nan {
                return TerminationStatus::Terminated(TerminationReason::SolverExit(
                    "hessian is NaN or infinite".into(),
                ));
            }
        }

        if let (Some(g), Some(h)) = (state.get_gradient(), state.get_hessian()) {
            let step = h * g;
            let norm = step.norm();
            if norm < self.step_size_tolerance() {
                return TerminationStatus::Terminated(TerminationReason::SolverExit(
                    "step size tolerance reached".into(),
                ));
            }
        }

        if state.get_cost() != state.get_prev_cost()
            && nalgebra::ComplexField::abs(state.get_cost() - state.get_prev_cost())
                < self.cost_tolerance()
        {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }

        TerminationStatus::NotTerminated
    }
}