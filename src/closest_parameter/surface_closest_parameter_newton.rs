use argmin::{argmin_error, argmin_error_closure, core::*, float};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimNameDiff, DimNameSub, Matrix2, OVector, Vector2, U1,
};

use crate::misc::FloatingPoint;

/// Customized Newton's method for finding the closest parameter on a NURBS surface
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct SurfaceClosestParameterNewton<F, D> {
    /// gamma
    gamma: F,
    /// domain of the parameter
    knot_domain: ((F, F), (F, F)),
    /// the target curve is closed or not
    closed: (bool, bool),
    phantom: std::marker::PhantomData<D>,
}

impl<F, D> SurfaceClosestParameterNewton<F, D>
where
    F: ArgminFloat + Clone,
{
    /// Construct a new instance of [`Newton`]
    pub fn new(domain: ((F, F), (F, F)), closed: (bool, bool)) -> Self {
        SurfaceClosestParameterNewton {
            // gamma: float!(0.25),
            gamma: float!(0.25),
            knot_domain: domain,
            closed,
            phantom: Default::default(),
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

impl<O, F, D> Solver<O, IterState<Vector2<F>, Vector2<F>, (), (), (), F>>
    for SurfaceClosestParameterNewton<F, D>
where
    F: FloatingPoint + ArgminFloat,
    O: CostFunction<Param = Vector2<F>, Output = F>
        + Gradient<Param = Vector2<F>, Gradient = OVector<F, DimNameDiff<D, U1>>>
        + Hessian<Param = Vector2<F>, Hessian = Vec<Vec<OVector<F, DimNameDiff<D, U1>>>>>,
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    const NAME: &'static str = "Closest parameter newton method";

    fn init(
        &mut self,
        problem: &mut Problem<O>,
        state: IterState<Vector2<F>, Vector2<F>, (), (), (), F>,
    ) -> Result<(IterState<Vector2<F>, Vector2<F>, (), (), (), F>, Option<KV>), Error> {
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
        state: IterState<Vector2<F>, Vector2<F>, (), (), (), F>,
    ) -> Result<(IterState<Vector2<F>, Vector2<F>, (), (), (), F>, Option<KV>), Error> {
        let param = state.get_param().ok_or_else(argmin_error_closure!(
            NotInitialized,
            concat!(
                "`Newton` requires an initial parameter vector. ",
                "Please provide an initial guess via `Executor`s `configure` method."
            )
        ))?;

        // let p = *param;
        // return Ok((state.param(p), None));

        let dif = problem.gradient(param)?;
        let e = problem.hessian(param)?;

        let s_u = &e[1][0];
        let s_v = &e[0][1];
        let s_uu = &e[2][0];
        let s_vv = &e[0][2];
        let s_uv = &e[1][1];
        // let s_vu = &e[1][1];

        let grad = Vector2::new(s_u.dot(&dif), s_v.dot(&dif));

        let distance = dif.norm();
        let eps = F::from_f64(1e-5).unwrap();

        // halt if point is close enough
        if distance < eps {
            let p = *param;
            // println!("halt");
            return Ok((state.param(p), None));
        }

        let u_d = s_uu.norm() < F::default_epsilon();
        let v_d = s_vv.norm() < F::default_epsilon();
        // println!("u_d: {}, v_d: {}", u_d, v_d);
        let new_param = match (u_d, v_d) {
            (false, false) => {
                let j00 = s_u.dot(s_u) + s_uu.dot(&dif);
                let j01 = s_u.dot(s_v) + s_uv.dot(&dif);
                let j11 = s_v.dot(s_v) + s_vv.dot(&dif);
                let hessian = Matrix2::new(j00, j01, j01, j11);
                let delta = hessian
                    .lu()
                    .solve(&-grad)
                    .ok_or(anyhow::anyhow!("Failed to solve"))?;
                *param + delta * self.gamma
            }
            (true, false) => {
                let v_delta = -grad.y / (s_v.dot(s_v) + s_vv.dot(&dif));
                *param + Vector2::new(F::zero(), v_delta) * self.gamma
            }
            (false, true) => {
                let u_delta = -grad.x / (s_u.dot(s_u) + s_uu.dot(&dif));
                *param + Vector2::new(u_delta, F::zero()) * self.gamma
            }
            _ => {
                return Err(anyhow::anyhow!("Invalid case"));
            }
        };

        /*
        let inv = hessian
            .try_inverse()
            .ok_or(anyhow::anyhow!("Failed to compute inverse matrix"))?;
        let delta = -inv * grad;
        let new_param = *param + delta * self.gamma;
        */

        // Constrain the parameter to the domain
        let new_param = Vector2::new(
            constrain(new_param.x, self.knot_domain.0, self.closed.0),
            constrain(new_param.y, self.knot_domain.1, self.closed.1),
        );

        let new_cost = problem.cost(&new_param)?;
        // println!("prev: {}, next: {}", state.get_cost(), new_cost);

        // halt if cost is not decreasing
        if state.get_cost() < new_cost {
            let p = *param;
            Ok((state.param(p), None))
        } else {
            Ok((state.cost(new_cost).param(new_param), None))
        }
    }

    fn terminate(
        &mut self,
        state: &IterState<Vector2<F>, Vector2<F>, (), (), (), F>,
    ) -> TerminationStatus {
        if state.iter > state.max_iters {
            return TerminationStatus::Terminated(TerminationReason::MaxItersReached);
        }

        match (state.get_param(), state.get_prev_param()) {
            (Some(current_param), Some(prev_param)) => {
                let delta = (current_param - prev_param).norm();
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

fn constrain<T: FloatingPoint>(parameter: T, domain: (T, T), closed: bool) -> T {
    if parameter < domain.0 {
        if closed {
            domain.1 - (parameter - domain.0)
        } else {
            domain.0 + T::default_epsilon()
        }
    } else if parameter > domain.1 {
        if closed {
            domain.0 + (parameter - domain.1)
        } else {
            domain.1 - T::default_epsilon()
        }
    } else {
        parameter
    }
}
