use argmin::{argmin_error, argmin_error_closure, core::*, float};
use argmin_math::{ArgminDot, ArgminInv, ArgminScaledSub};

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

impl<'a, O, P, G, H, F> Solver<O, IterState<P, G, (), H, (), F>> for ClosestParameterNewton<F, P>
where
    O: Gradient<Param = P, Gradient = G> + Hessian<Param = P, Hessian = H>,
    P: Clone + ArgminScaledSub<P, F, P> + ArgminFloat,
    H: ArgminInv<H> + ArgminDot<G, P>,
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
        let new_param = param.scaled_sub(&self.gamma, &hessian.inv()?.dot(&grad));

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

    fn terminate(&mut self, state: &IterState<P, G, (), H, (), F>) -> TerminationStatus {
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
    }
}
