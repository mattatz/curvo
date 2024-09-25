use argmin::{argmin_error_closure, core::*};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimNameDiff, DimNameSub, Matrix2, OVector, Vector2, U1,
};

use crate::misc::FloatingPoint;

/// Customized Newton's method for finding the closest parameter on a NURBS surface
/// Original source: https://argmin-rs.github.io/argmin/argmin/solver/newton/struct.Newton.html
#[derive(Clone, Copy)]
pub struct SurfaceClosestParameterNewton<T, D> {
    /// domain of the parameter
    knot_domain: ((T, T), (T, T)),
    /// the target curve is closed or not
    closed: (bool, bool),
    phantom: std::marker::PhantomData<D>,
}

impl<T, D> SurfaceClosestParameterNewton<T, D>
where
    T: ArgminFloat + Clone,
{
    /// Construct a new instance of [`Newton`]
    pub fn new(domain: ((T, T), (T, T)), closed: (bool, bool)) -> Self {
        SurfaceClosestParameterNewton {
            knot_domain: domain,
            closed,
            phantom: Default::default(),
        }
    }
}

impl<O, F, D> Solver<O, IterState<Vector2<F>, Vector2<F>, (), (), (), F>>
    for SurfaceClosestParameterNewton<F, D>
where
    F: FloatingPoint + ArgminFloat,
    O: Gradient<Param = Vector2<F>, Gradient = OVector<F, DimNameDiff<D, U1>>>
        + Hessian<Param = Vector2<F>, Hessian = Vec<Vec<OVector<F, DimNameDiff<D, U1>>>>>,
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    const NAME: &'static str = "Closest parameter newton method";

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

        // point coincidence
        // |S(u,v) - p| < epsilon
        let distance = dif.norm();

        // cosine
        // |Su(u,v) * (S(u,v) - p)|
        // ------------------------ < epsilon
        // |Su(u,v)| |S(u,v) - p|
        let f = s_u.dot(&dif);
        let d1 = s_u.norm() * distance;
        let c1 = f / d1;

        // |Sv(u,v) * (S(u,v) - p)|
        // ------------------------ < epsilon
        // |Sv(u,v)| |S(u,v) - p|
        let g = s_v.dot(&dif);
        let d2 = s_v.norm() * distance;
        let c2 = g / d2;

        let eps = F::from_f64(1e-4).unwrap();

        // halt if the conditions are met
        if distance < eps && c1 < eps && c2 < eps {
            let p = *param;
            return Ok((state.param(p), None));
        }

        let j00 = s_u.dot(s_u) + s_uu.dot(&dif);
        let j01 = s_u.dot(s_v) + s_uv.dot(&dif);
        let j10 = j01;
        let j11 = s_v.dot(s_v) + s_vv.dot(&dif);
        let jacobian = Matrix2::new(j00, j01, j10, j11);
        let k = Vector2::new(-f, -g);
        let d = jacobian
            .lu()
            .solve(&k)
            .ok_or(anyhow::anyhow!("Failed to solve jacobian"))?;

        let new_param = d + param;

        // Constrain the parameter to the domain
        let new_param = Vector2::new(
            constrain(new_param.x, self.knot_domain.0, self.closed.0),
            constrain(new_param.y, self.knot_domain.1, self.closed.1),
        );

        Ok((state.param(new_param), None))
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
