use argmin::{argmin_error, argmin_error_closure, core::*, float};
use argmin_math::ArgminScaledSub;

/// Customized Newton's method for finding the closest parameter on a NURBS curve
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct ClosestParameterNewton<F, P> {
    /// gamma
    gamma: F,
    /// domain of the parameter
    knot_domain: (P, P),
    /// the target curve is closed or not
    closed: bool,
}

impl<F, P> ClosestParameterNewton<F, P>
where
    F: ArgminFloat,
    P: Clone + ArgminScaledSub<P, F, P>,
{
    /// Construct a new instance of [`Newton`]
    pub fn new(domain: (P, P), closed: bool) -> Self {
        ClosestParameterNewton {
            gamma: float!(1.0),
            knot_domain: domain,
            closed,
        }
    }

    /// Set step size gamma
    ///
    /// Gamma must be in `(0, 1]` and defaults to `1`.
    #[allow(unused)]
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

impl<O, F> Solver<O, IterState<F, F, (), F, (), F>> for ClosestParameterNewton<F, F>
where
    O: Gradient<Param = F, Gradient = F> + Hessian<Param = F, Hessian = F>,
    F: Clone + ArgminScaledSub<F, F, F> + ArgminFloat,
    F: ArgminFloat,
{
    const NAME: &'static str = "Closest parameter newton method";

    fn next_iter(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<F, F, (), F, (), F>,
    ) -> Result<(IterState<F, F, (), F, (), F>, Option<KV>), Error> {
        let param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        let grad = problem.gradient(param)?;
        let hessian = problem.hessian(param)?;
        let inv = F::one() / hessian;
        let new_param = param.scaled_sub(&self.gamma, &(inv * grad));

        // Constrain the parameter to the domain
        let new_param = if new_param < self.knot_domain.0 {
            if self.closed {
                self.knot_domain.1 - (new_param - self.knot_domain.0)
            } else {
                self.knot_domain.0
            }
        } else if new_param > self.knot_domain.1 {
            if self.closed {
                self.knot_domain.0 + (new_param - self.knot_domain.1)
            } else {
                self.knot_domain.1
            }
        } else {
            new_param
        };

        Ok((state.param(new_param), None))
    }

    fn terminate(&mut self, state: &IterState<F, F, (), F, (), F>) -> TerminationStatus {
        if state.iter > state.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }

        match (state.get_param(), state.get_prev_param()) {
            (Some(current_param), Some(prev_param)) => {
                let delta = (*current_param - *prev_param).abs();
                if delta < F::epsilon() {
                    TerminationStatus::Terminated(TerminationReason::SolverConverged)
                } else {
                    TerminationStatus::NotTerminated
                }
            }
            _ => TerminationStatus::NotTerminated,
        }
    }
}
