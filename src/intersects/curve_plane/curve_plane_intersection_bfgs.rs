use argmin::{argmin_error_closure, core::*};
use nalgebra::{Matrix1, Vector1};

use crate::{misc::FloatingPoint, prelude::CurveIntersectionBFGS};

type CurvePlaneIterState<F> = IterState<Vector1<F>, Vector1<F>, (), Matrix1<F>, (), F>;

impl<O, F> Solver<O, CurvePlaneIterState<F>> for CurveIntersectionBFGS<F>
where
    O: Gradient<Param = Vector1<F>, Gradient = Vector1<F>>
        + CostFunction<Param = Vector1<F>, Output = F>,
    F: FloatingPoint + ArgminFloat,
{
    const NAME: &'static str = "Curve plane intersection newton method with line search";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: CurvePlaneIterState<F>,
    ) -> Result<(CurvePlaneIterState<F>, Option<KV>), Error> {
        let x0 = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`CurvePlaneIntersectionBFGS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;
        let cost = problem.cost(x0)?;

        Ok((state.cost(cost), None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: CurvePlaneIterState<F>,
    ) -> Result<(CurvePlaneIterState<F>, Option<KV>), Error> {
        let x0 = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`CurvePlaneIntersectionBFGS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let f0 = state.get_cost();

        let g0 = match state.get_gradient() {
            Some(prev) => *prev,
            None => problem.gradient(x0)?,
        };

        let h0 = state.get_hessian().cloned().unwrap_or(Matrix1::identity());

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

    fn terminate(&mut self, state: &CurvePlaneIterState<F>) -> TerminationStatus {
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
