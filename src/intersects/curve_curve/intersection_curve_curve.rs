use std::cmp::Ordering;

use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Matrix2, OPoint,
    Point2, Vector2, U1,
};
use num_traits::Float;

use crate::{
    curve::NurbsCurve,
    knot::KnotVector,
    misc::FloatingPoint,
    prelude::{
        curve_curve::helper::{find_line_string_intersection, to_line_string_helper},
        BoundingBoxTraversal, CurveBoundingBoxTree, HasIntersection, Intersects,
    },
};

use super::{
    CurveCurveIntersection, CurveIntersectionBFGS, CurveIntersectionProblem,
    CurveIntersectionSolverOptions,
};

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for NurbsCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<CurveCurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points with another curve by gauss-newton line search
    /// * `other` - The other curve to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Point3, Vector2};
    /// use approx::assert_relative_eq;
    /// let unit_circle = NurbsCurve2D::try_circle(
    ///     &Point2::origin(),
    ///     &Vector2::x(),
    ///     &Vector2::y(),
    ///     1.
    /// ).unwrap();
    /// let line = NurbsCurve2D::try_new(
    ///     1,
    ///     vec![
    ///         Point3::new(-2.0, 0.0, 1.),
    ///         Point3::new(2.0, 0.0, 1.),
    ///     ],
    ///     vec![0., 0., 1., 1.],
    /// ).unwrap();
    ///
    /// // Hyperparameters for the intersection solver
    /// let options = CurveIntersectionSolverOptions {
    ///     minimum_distance: 1e-5, // minimum distance between intersections
    ///     cost_tolerance: 1e-12, // cost tolerance for the solver convergence
    ///     max_iters: 200, // maximum number of iterations in the solver
    ///     ..Default::default()
    /// };
    ///
    /// let mut intersections = unit_circle.find_intersection(&line, Some(options)).unwrap();
    /// assert_eq!(intersections.len(), 2);
    ///
    /// intersections.sort_by(|i0, i1| {
    ///     i0.a().0.x.partial_cmp(&i1.a().0.x).unwrap()
    /// });
    /// let p0 = &intersections[0];
    /// assert_relative_eq!(p0.a().0, Point2::new(-1.0, 0.0), epsilon = 1e-5);
    /// let p1 = &intersections[1];
    /// assert_relative_eq!(p1.a().0, Point2::new(1.0, 0.0), epsilon = 1e-5);
    /// ```
    fn find_intersection(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        option: Self::Option,
    ) -> Self::Output {
        if self.degree() == 1 && other.degree() == 1 && D::dim() == 3 {
            // 2d polyline intersection
            let p0 = self
                .dehomogenized_control_points()
                .iter()
                .map(|p| Point2::from_slice(p.coords.as_slice()))
                .collect_vec();
            let p1 = other
                .dehomogenized_control_points()
                .iter()
                .map(|p| Point2::from_slice(p.coords.as_slice()))
                .collect_vec();
            let l0 = to_line_string_helper(&p0);
            let l1 = to_line_string_helper(&p1);
            let intersections = find_line_string_intersection(&l0, &l1)?;

            let find_parameter = |points: &Vec<Point2<T>>,
                                  knots: &KnotVector<T>,
                                  point: Point2<f64>,
                                  index: usize|
             -> anyhow::Result<T> {
                anyhow::ensure!(index + 1 < points.len(), "index out of bounds");

                let prev = points[index];
                let next = points[index + 1];
                let k0 = knots[index + 1];
                let k1 = knots[index + 2];

                let d = (next - prev).norm();
                let d2 = (next - point.map(|x| T::from_f64(x).unwrap())).norm();
                let t = T::one() - d2 / d;

                Ok(k0 + t * (k1 - k0))
            };

            let its = intersections
                .into_iter()
                .map(|it| {
                    let pt = it.point();
                    let (i0, i1) = it.line_index();
                    let t0 = find_parameter(&p0, self.knots(), pt, i0)?;
                    let t1 = find_parameter(&p1, other.knots(), pt, i1)?;
                    let pt = OPoint::<T, DimNameDiff<D, U1>>::from_slice(
                        &pt.coords
                            .as_slice()
                            .iter()
                            .map(|x| T::from_f64(*x).unwrap())
                            .collect_vec(),
                    );
                    Ok(CurveCurveIntersection::new((pt.clone(), t0), (pt, t1)))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            return Ok(group_and_extract_closest_intersections(its));
        }

        let options = option.unwrap_or_default();

        let ta = CurveBoundingBoxTree::new(
            self,
            Some(
                self.knots_domain_interval() / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );
        let tb = CurveBoundingBoxTree::new(
            other,
            Some(
                other.knots_domain_interval()
                    / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );

        let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;

        let a_domain = self.knots_domain();
        let b_domain = other.knots_domain();

        let intersections = traversed
            .into_pairs_iter()
            .filter_map(|(a, b)| {
                let ca = a.curve_owned();
                let cb = b.curve_owned();

                let problem = CurveIntersectionProblem::new(&ca, &cb);

                // let inv = T::from_f64(0.5).unwrap();
                // let d0 = ca.knots_domain();
                // let d1 = cb.knots_domain();

                // Define initial parameter vector
                let init_param = Vector2::<T>::new(
                    ca.knots_domain().0,
                    cb.knots_domain().0,
                    // (d0.0 + d0.1) * inv,
                    // (d1.0 + d1.1) * inv,
                );

                // Set up solver
                let solver = CurveIntersectionBFGS::<T>::new()
                    .with_step_size_tolerance(options.step_size_tolerance)
                    .with_cost_tolerance(options.cost_tolerance);

                // Run solver
                let res = Executor::new(problem, solver)
                    .configure(|state| {
                        state
                            .param(init_param)
                            .inv_hessian(Matrix2::identity())
                            .max_iters(options.max_iters)
                    })
                    .run();

                match res {
                    Ok(r) => {
                        // println!("{}", r.state().get_termination_status());
                        r.state().get_best_param().and_then(|param| {
                            if (a_domain.0..=a_domain.1).contains(&param[0])
                                && (b_domain.0..=b_domain.1).contains(&param[1])
                            {
                                let p0 = self.point_at(param[0]);
                                let p1 = other.point_at(param[1]);
                                Some(CurveCurveIntersection::new((p0, param[0]), (p1, param[1])))
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
                // filter out intersections that are too far away
                let p0 = &it.a().0;
                let p1 = &it.b().0;
                let d = (p0 - p1).norm();
                d < options.minimum_distance
            })
            .collect_vec();

        let pts = group_and_extract_closest_intersections(intersections);
        Ok(pts)
    }
}

/// Group intersections by parameter and extract the closest intersection in each group
fn group_and_extract_closest_intersections<T, D>(
    intersections: Vec<CurveCurveIntersection<OPoint<T, D>, T>>,
) -> Vec<CurveCurveIntersection<OPoint<T, D>, T>>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName,
    DefaultAllocator: Allocator<D>,
{
    let sorted = intersections
        .into_iter()
        .sorted_by(|x, y| x.a().1.partial_cmp(&y.a().1).unwrap_or(Ordering::Equal))
        .collect_vec();

    let parameter_minimum_distance = T::from_f64(1e-3).unwrap();
    let groups = sorted
        .into_iter()
        .map(|pt| vec![pt])
        .coalesce(|x, y| {
            let x0 = &x[x.len() - 1];
            let y0 = &y[y.len() - 1];
            let da = Float::abs(x0.a().1 - y0.a().1);
            let db = Float::abs(x0.b().1 - y0.b().1);
            if da < parameter_minimum_distance || db < parameter_minimum_distance {
                // merge near parameter results
                let group = [x, y].concat();
                Ok(group)
            } else {
                Err((x, y))
            }
        })
        .collect::<Vec<Vec<CurveCurveIntersection<OPoint<T, D>, T>>>>()
        .into_iter()
        .collect_vec();

    groups
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
        .collect_vec()
}
