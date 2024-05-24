use argmin::{argmin_error, argmin_error_closure, core::*, float};
use argmin_math::ArgminDot;
use nalgebra::{Matrix2, Vector2};

use crate::misc::FloatingPoint;

/// Customized Newton's method for finding the intersections between NURBS curves
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct CurveIntersectionNewton<F> {
    /// gamma
    gamma: F,

    /// tolerance
    tolerance: F,

    /// maximum number of iterations for line search
    line_search_max_iters: u64,
}

impl<F> Default for CurveIntersectionNewton<F>
where
    F: FloatingPoint,
{
    fn default() -> Self {
        Self {
            gamma: float!(1.0),
            tolerance: float!(1e-8),
            line_search_max_iters: 100,
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

    /// Set step size gamma
    ///
    /// Gamma must be in `(0, 1]` and defaults to `1`.
    pub fn with_gamma(mut self, gamma: F) -> Result<Self, Error> {
        if gamma <= float!(0.0) || gamma > float!(1.0) {
            return Err(argmin_error!(
                InvalidParameter,
                "Newton: gamma must be in  (0, 1]."
            ));
        }
        self.gamma = gamma;
        Ok(self)
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
        let df0 = g0.dot(&step) * F::from_f64(1e-1).unwrap();
        let mut x1 = *x0;
        let mut f1 = f0;

        let dec = F::from_f64(0.5).unwrap();
        for _ in 0..self.line_search_max_iters {
            if t * norm < self.tolerance {
                break;
            }
            let s = step * t;
            x1 = x0 + s;
            f1 = problem.cost(&x1)?;
            if f1 - f0 >= t * df0 {
                t *= dec;
            } else {
                break;
            }
        }

        // println!("{}: {}, {}", state.iter, x1, f1);

        let g1 = problem.gradient(&x1)?;
        let y = g1 - g0;
        let s = step * t;
        let ys = y.dot(&s);
        let s_t = tensor(&s, &s);
        let hy = h0 * y;

        let h1 = (h0 + s_t * ((ys + y.dot(&hy)) / (ys * ys)))
            - ((tensor(&hy, &s) + tensor(&s, &hy)) / ys);

        Ok((state.param(x1).cost(f1).gradient(g1).hessian(h1), None))
    }

    fn terminate(
        &mut self,
        state: &IterState<Vector2<F>, Vector2<F>, (), Matrix2<F>, (), F>,
    ) -> TerminationStatus {
        if state.iter > state.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }

        if let (Some(g), Some(h)) = (state.get_gradient(), state.get_hessian()) {
            let step = h * g;
            let norm = step.norm();
            if norm < self.tolerance {
                return TerminationStatus::Terminated(TerminationReason::SolverConverged);
            }
        }

        if nalgebra::ComplexField::abs(state.get_cost() - state.get_prev_cost()) < self.tolerance {
            return TerminationStatus::Terminated(TerminationReason::SolverConverged);
        }

        TerminationStatus::NotTerminated
    }
}

fn tensor<T: FloatingPoint>(a: &Vector2<T>, b: &Vector2<T>) -> Matrix2<T> {
    let mut res = Matrix2::zeros();
    res[(0, 0)] = a[0] * b[0];
    res[(0, 1)] = a[0] * b[1];
    res[(1, 0)] = a[1] * b[0];
    res[(1, 1)] = a[1] * b[1];
    res
}
