use std::cmp::Ordering;

use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Matrix3, OPoint,
    Vector2, Vector3, U1,
};
use num_traits::Float;

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{
        BoundingBoxTraversal, CurveBoundingBoxTree, CurveIntersectionSolverOptions,
        HasIntersection, Intersects, SurfaceBoundingBoxTree, SurfaceCurveIntersection,
    },
    surface::{self, NurbsSurface, UVDirection},
};

use super::{SurfaceCurveIntersectionBFGS, SurfaceCurveIntersectionProblem};

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for NurbsSurface<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<SurfaceCurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    ///
    #[allow(clippy::type_complexity)]
    fn find_intersections(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        option: Self::Option,
    ) -> Self::Output {
        let options = option.unwrap_or_default();

        let div = T::one() / T::from_usize(options.knot_domain_division).unwrap();
        let interval = self.knots_domain_interval();
        let ta = SurfaceBoundingBoxTree::new(
            self,
            UVDirection::U,
            Some((interval.0 * div, interval.1 * div)),
        );
        let tb = CurveBoundingBoxTree::new(other, Some(other.knots_domain_interval() * div));

        let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;
        let (surface_u_domain, surface_v_domain) = self.knots_domain();
        let curve_domain = other.knots_domain();

        let intersections = traversed
            .into_pairs_iter()
            .filter_map(|(a, b)| {
                let surface = a.surface_owned();
                let curve = b.curve_owned();

                let div = T::from_f64(0.5).unwrap();

                let d = curve.knots_domain();
                let curve_parameter = (d.0 + d.1) * div;

                let (u, v) = surface.knots_domain();
                let surface_parameter = ((u.0 + u.1) * div, (v.0 + v.1) * div);

                let problem = SurfaceCurveIntersectionProblem::new(&surface, &curve);

                // Define initial parameter vector
                let init_param =
                    Vector3::new(curve_parameter, surface_parameter.0, surface_parameter.1);

                // Set up solver
                let solver = SurfaceCurveIntersectionBFGS::<T>::new()
                    .with_step_size_tolerance(options.step_size_tolerance)
                    .with_cost_tolerance(options.cost_tolerance);

                // Run solver
                let res = Executor::new(problem, solver)
                    .configure(|state| {
                        state
                            .param(init_param)
                            .inv_hessian(Matrix3::identity())
                            .max_iters(options.max_iters)
                    })
                    .run();

                match res {
                    Ok(r) => {
                        // println!("{}", r.state().get_termination_status());
                        r.state().get_best_param().and_then(|param| {
                            if (surface_u_domain.0..=surface_u_domain.1).contains(&param.y)
                                && (surface_v_domain.0..=surface_v_domain.1).contains(&param.z)
                                && (curve_domain.0..=curve_domain.1).contains(&param.x)
                            {
                                let p0 = self.point_at(param.y, param.z);
                                let p1 = other.point_at(param.x);
                                Some(SurfaceCurveIntersection::new(
                                    (p0, (param.y, param.z)),
                                    (p1, param.x),
                                ))
                            } else {
                                None
                            }
                        })
                    }
                    Err(_e) => {
                        // println!("{}", e);
                        None
                    }
                }
            })
            .filter(|it| {
                // filter out intersections that are too close
                let p0 = &it.a().0;
                let p1 = &it.b().0;
                let d = (p0 - p1).norm();
                d < options.minimum_distance
            })
            .collect_vec();

        let sorted = intersections
            .into_iter()
            .sorted_by(|x, y| x.b().1.partial_cmp(&y.b().1).unwrap_or(Ordering::Equal))
            .collect_vec();

        // group near parameter results & extract the closest one in each group
        let parameter_minimum_distance = T::from_f64(1e-3).unwrap();
        let groups = sorted
            .into_iter()
            .map(|pt| vec![pt])
            .coalesce(|x, y| {
                let x0 = &x[x.len() - 1];
                let y0 = &y[y.len() - 1];
                let xs = x0.a().1;
                let ys = y0.a().1;
                let da0 = Float::abs(xs.0 - ys.0);
                let da1 = Float::abs(xs.1 - ys.1);
                let db = Float::abs(x0.b().1 - y0.b().1);
                if da0 < parameter_minimum_distance
                    || da1 < parameter_minimum_distance
                    || db < parameter_minimum_distance
                {
                    // merge near parameter results
                    let group = [x, y].concat();
                    Ok(group)
                } else {
                    Err((x, y))
                }
            })
            .collect::<Vec<Vec<_>>>()
            .into_iter()
            .collect_vec();

        let pts = groups
            .into_iter()
            .filter_map(|group| match group.len() {
                1 => Some(group[0].clone()),
                _ => {
                    // find the closest intersection in the group
                    group
                        .iter()
                        .map(|it| {
                            let delta = &it.a().0 - &it.b().0;
                            let norm = delta.norm_squared();
                            (it, norm)
                        })
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                        .map(|closest| closest.0.clone())
                }
            })
            .collect_vec();

        Ok(pts)
    }
}
