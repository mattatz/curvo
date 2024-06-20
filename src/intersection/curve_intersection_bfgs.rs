use anyhow::Error;
use argmin::{
    argmin_error, argmin_error_closure,
    core::{
        ArgminFloat, CostFunction, Executor, Gradient, IterState, LineSearch, OptimizationResult,
        Problem, Solver, State, TerminationReason, TerminationStatus, KV,
    },
    float,
};
use nalgebra::{ComplexField, Matrix2, Vector2};

use crate::misc::FloatingPoint;

/// Customized quasi-Newton's method for finding the intersections between NURBS curves
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/quasinewton/struct.BFGS.html
#[derive(Clone)]
pub struct CurveIntersectionBFGS<L, F> {
    /// line search
    linesearch: L,
    /// Tolerance for the stopping criterion based on the change of the norm on the gradient
    tol_grad: F,
    /// Tolerance for the stopping criterion based on the change of the cost stopping criterion
    tol_cost: F,
}

impl<L, F> CurveIntersectionBFGS<L, F>
where
    F: FloatingPoint,
{
    pub fn new(linesearch: L) -> Self {
        CurveIntersectionBFGS {
            linesearch,
            tol_grad: F::default_epsilon().sqrt(),
            tol_cost: F::default_epsilon(),
        }
    }

    pub fn with_tolerance_grad(mut self, tol_grad: F) -> Result<Self, Error> {
        if tol_grad < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`BFGS`: gradient tolerance must be >= 0."
            ));
        }
        self.tol_grad = tol_grad;
        Ok(self)
    }

    pub fn with_tolerance_cost(mut self, tol_cost: F) -> Result<Self, Error> {
        if tol_cost < float!(0.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "`BFGS`: cost tolerance must be >= 0."
            ));
        }
        self.tol_cost = tol_cost;
        Ok(self)
    }
}

impl<O, L, F> Solver<O, IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>>
    for CurveIntersectionBFGS<L, F>
where
    O: CostFunction<Param = Vector2<F>, Output = F>
        + Gradient<Param = Vector2<F>, Gradient = Vector2<F>>,
    L: Clone
        + LineSearch<Vector2<F>, F>
        + Solver<O, IterState<Vector2<F>, Vector2<F>, (), (), (), F>>,
    F: FloatingPoint + ArgminFloat,
{
    const NAME: &'static str = "Curve intersection quasi-newton method";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> Result<
        (
            IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
            Option<KV>,
        ),
        Error,
    > {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`BFGS` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`BFGS` requires an initial inverse Hessian. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let cost = state.get_cost();
        let cost = if cost.is_infinite() {
            problem.cost(&param)?
        } else {
            cost
        };

        let grad = state
            .take_gradient()
            .map(Result::Ok)
            .unwrap_or_else(|| problem.gradient(&param))?;

        Ok((
            state
                .param(param)
                .cost(cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            None,
        ))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        mut state: IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> Result<
        (
            IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
            Option<KV>,
        ),
        Error,
    > {
        let param = state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`BFGS`: Parameter vector in state not set."
        ))?;

        let cur_cost = state.get_cost();

        let prev_grad = state.take_gradient().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`BFGS`: Gradient in state not set."
        ))?;

        let inv_hessian = state.take_inv_hessian().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`BFGS`: Inverse Hessian in state not set."
        ))?;

        let p = -inv_hessian * prev_grad;

        self.linesearch.search_direction(p);

        // println!("iter: {:?}, p: {:?}", state.get_iter(), param);

        // Run solver
        let solver = Executor::new(problem.take_problem().unwrap(), self.linesearch.clone())
            .configure(|config| {
                config
                    .param(param.clone())
                    .gradient(prev_grad.clone())
                    .cost(cur_cost)
                    .max_iters(32)
            })
            .ctrlc(false);

        let OptimizationResult {
            problem: line_problem,
            state: mut sub_state,
            ..
        } = solver.run()?;

        let xk1 = sub_state.take_param().ok_or_else(argmin_error_closure!(
            PotentialBug,
            "`BFGS`: No parameters returned by line search."
        ))?;

        let next_cost = sub_state.get_cost();

        // take care of function eval counts
        problem.consume_problem(line_problem);

        let grad = problem.gradient(&xk1)?;

        let yk = grad - prev_grad;

        let sk = xk1 - param;

        let yksk = yk.dot(&sk);
        let rhok = float!(1.0) / yksk;

        let e = Matrix2::identity();
        let mat1 = sk * yk.transpose();
        let mat1 = mat1 * rhok;

        let tmp1 = e - mat1;

        let mat2 = mat1.transpose();
        let tmp2 = e - mat2;

        let sksk = sk * sk.transpose();
        let sksk = sksk * rhok;

        // if state.get_iter() == 0 {
        //     let ykyk: f64 = yk.dot(&yk);
        //     self.inv_hessian = self.inv_hessian.eye_like().mul(&(yksk / ykyk));
        //     println!("{:?}", self.inv_hessian);
        // }

        let inv_hessian = tmp1 * (inv_hessian.dot(&tmp2)) + sksk;

        Ok((
            state
                .param(xk1)
                .cost(next_cost)
                .gradient(grad)
                .inv_hessian(inv_hessian),
            None,
        ))
    }

    fn terminate(
        &mut self,
        state: &IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> TerminationStatus {
        if state.get_gradient().unwrap().norm() < self.tol_grad {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        if ComplexField::abs(state.get_prev_cost() - state.cost) < self.tol_cost {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }
        TerminationStatus::NotTerminated
    }
}
