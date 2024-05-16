use argmin::{argmin_error, argmin_error_closure, core::*, float};
use argmin_math::{ArgminInv, ArgminMul, ArgminScaledSub};

/// Customized Newton's method for finding the intersections between NURBS curves
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct CurveIntersectionNewton<F> {
    /// gamma
    gamma: F,
}

impl<F> Default for CurveIntersectionNewton<F>
where
    F: ArgminFloat,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> CurveIntersectionNewton<F>
where
    F: ArgminFloat,
{
    /// Construct a new instance of [`Newton`]
    pub fn new() -> Self {
        CurveIntersectionNewton { gamma: float!(1.0) }
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

impl<'a, O, P, G, H, F> Solver<O, IterState<P, G, (), H, (), F>> for CurveIntersectionNewton<F>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone + ArgminScaledSub<P, F, P> + ArgminFloat,
    H: ArgminInv<H> + ArgminMul<G, P>,
    F: ArgminFloat,
{
    const NAME: &'static str = "Newton method";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<P, G, (), H, (), F>,
    ) -> Result<(IterState<P, G, (), H, (), F>, Option<KV>), Error> {
        let param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let grad = problem.gradient(param)?;
        let hessian = problem.hessian(param)?;
        let new_param = param.scaled_sub(&self.gamma, &hessian.inv()?.mul(&grad));

        Ok((state.param(new_param), None))
    }

    fn terminate(&mut self, _state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
        todo!()
        /*
        match (state.get_param(), state.get_prev_param()) {
            (Some(current_param), Some(prev_param)) => {
                let delta = (*current_param - *prev_param).abs();
                if delta < P::epsilon() {
                    TerminationStatus::Terminated(TerminationReason::SolverConverged)
                } else {
                    TerminationStatus::NotTerminated
                }
            }
            _ => TerminationStatus::NotTerminated,
        }
        */
    }
}
