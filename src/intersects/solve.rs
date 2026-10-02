//! What the intersection finders share: the scales the solver works in, and the choice of the
//! intersections among what it finds.

use std::cmp::Ordering;

use argmin::core::{
    ArgminFloat, CostFunction, Error, Executor, Gradient, IterState, Solver, State,
};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, SMatrix,
    SVector, U1,
};

use crate::{
    bounding_box::BoundingBox,
    curve::{nurbs_curve::dehomogenize, NurbsCurve},
    misc::FloatingPoint,
    surface::NurbsSurface,
};

/// A parameter of a problem: its domain, how far the geometry travels over it, and whether the
/// geometry is closed in it, the end of the domain being the same place as its start.
pub(crate) struct Parameter<T> {
    domain: (T, T),
    travel: T,
    closed: bool,
}

/// The scales of a problem with `N` parameters: the size of the geometry and its parameters.
///
/// The solver works on distances divided by the size, and on parameters scaled so that a step of
/// one travels about one size. Its tolerances, and the distance below which two points are an
/// intersection, are then relative: the same geometry at another size, or parameterized over
/// another domain, has the same intersections.
pub(crate) struct Scales<T, const N: usize> {
    size: T,
    start: SVector<T, N>,
    length: SVector<T, N>,
    /// What a step of one in the solver is in each parameter.
    unit: SVector<T, N>,
    closed: [bool; N],
}

impl<T: FloatingPoint, const N: usize> Scales<T, N> {
    /// The scales of geometry of the given `sizes` — the smallest one that is not zero counts, as
    /// an intersection is no larger than the smaller of two objects — with `parameters`.
    pub(crate) fn new(sizes: &[T], parameters: [Parameter<T>; N]) -> Self {
        let size = sizes
            .iter()
            .copied()
            .filter(|size| *size > T::zero())
            .reduce(|a, b| a.min(b))
            .unwrap_or(T::one());
        // a domain of no length is left as it is
        let length = SVector::from_fn(|i, _| {
            let (start, end) = parameters[i].domain;
            if end > start {
                end - start
            } else {
                T::one()
            }
        });
        Self {
            size,
            start: SVector::from_fn(|i, _| parameters[i].domain.0),
            length,
            // and so is a parameter the geometry does not move with
            unit: SVector::from_fn(|i, _| {
                if parameters[i].travel > T::zero() {
                    length[i] * size / parameters[i].travel
                } else {
                    length[i]
                }
            }),
            closed: parameters.map(|parameter| parameter.closed),
        }
    }

    /// The distance `relative` to the size of the geometry.
    pub(crate) fn distance(&self, relative: T) -> T {
        relative * self.size
    }

    /// Whether the geometry is closed in the parameter `index`.
    pub(crate) fn is_closed(&self, index: usize) -> bool {
        self.closed[index]
    }

    /// The value of the parameter `index` halfway between `x` and `y`, where two candidates for
    /// the same intersection are expected to be in contact too.
    ///
    /// On geometry closed in this parameter, values more than half the domain apart are on the
    /// two sides of the seam, and halfway between them is [`Scales::halfway_across`] it.
    pub(crate) fn halfway(&self, index: usize, x: T, y: T) -> T {
        if self.closed[index] && (x - y).abs() > self.length[index] * T::from_f64(0.5).unwrap() {
            self.halfway_across(index, x, y)
        } else {
            (x + y) * T::from_f64(0.5).unwrap()
        }
    }

    /// The value of the parameter `index` halfway between `x` and `y` going through the seam of
    /// geometry closed in it.
    pub(crate) fn halfway_across(&self, index: usize, x: T, y: T) -> T {
        let (start, length) = (self.start[index], self.length[index]);
        let across = (x + y + length) * T::from_f64(0.5).unwrap();
        if across > start + length {
            across - length
        } else {
            across
        }
    }

    fn normalize(&self, parameters: &SVector<T, N>) -> SVector<T, N> {
        (parameters - self.start).component_div(&self.unit)
    }

    fn denormalize(&self, normalized: &SVector<T, N>) -> SVector<T, N> {
        normalized.component_mul(&self.unit) + self.start
    }

    /// Run `solver` on `problem` in these scales from the parameters `init`, and return the
    /// parameters it stops at.
    pub(crate) fn solve<O, S>(
        &self,
        problem: O,
        solver: S,
        init: SVector<T, N>,
        max_iters: u64,
    ) -> Option<SVector<T, N>>
    where
        T: ArgminFloat,
        O: CostFunction<Param = SVector<T, N>, Output = T>
            + Gradient<Param = SVector<T, N>, Gradient = SVector<T, N>>,
        S: for<'a> Solver<Normalized<'a, O, T, N>, SolverState<T, N>>,
    {
        let normalized = Normalized {
            problem,
            scales: self,
        };
        let result = Executor::new(normalized, solver)
            .configure(|state| {
                state
                    .param(self.normalize(&init))
                    .inv_hessian(SMatrix::identity())
                    .max_iters(max_iters)
            })
            .run()
            .ok()?;
        let parameters = self.denormalize(result.state().get_best_param()?);
        // a solver that ran away has nothing to say
        parameters
            .iter()
            .all(|p| p.is_finite())
            .then_some(parameters)
    }
}

/// The size of a curve, the diagonal of its bounding box, and its parameter. The curve travels the
/// length of its control polygon, or about that.
pub(crate) fn curve_scale<T: FloatingPoint, D>(curve: &NurbsCurve<T, D>) -> (T, Parameter<T>)
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let size = BoundingBox::from(curve).size().norm();
    let domain = curve.knots_domain();
    let ends = curve.point_at(domain.0) - curve.point_at(domain.1);
    let parameter = Parameter {
        domain,
        travel: polygon_length(curve.control_points().iter()),
        closed: ends.norm() <= closing_distance(size),
    };
    (size, parameter)
}

/// The size of a surface, the diagonal of its bounding box, and its parameters u and v. The
/// surface travels the length of its longest control polygon in each direction, or about that.
pub(crate) fn surface_scale<T: FloatingPoint, D>(
    surface: &NurbsSurface<T, D>,
) -> (T, [Parameter<T>; 2])
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let size = BoundingBox::from(surface).size().norm();
    let control_points = surface.control_points();
    let longest = |lengths: &mut dyn Iterator<Item = T>| lengths.fold(T::zero(), |a, b| a.max(b));
    let columns = control_points.first().map_or(0, |row| row.len());
    let u_travel = longest(
        &mut (0..columns)
            .map(|j| polygon_length(control_points.iter().filter_map(move |row| row.get(j)))),
    );
    let v_travel = longest(&mut control_points.iter().map(|row| polygon_length(row.iter())));

    // Closed in a direction when the two ends of the domain are the same place all along the
    // other direction, which a few places along it are taken to show.
    let (u_domain, v_domain) = surface.knots_domain();
    let along = |(start, end): (T, T)| {
        (0..=4).map(move |i| start + (end - start) * T::from_f64(i as f64 / 4.).unwrap())
    };
    let closing = closing_distance(size);
    let u_closed = along(v_domain).all(|v| {
        (surface.point_at(u_domain.0, v) - surface.point_at(u_domain.1, v)).norm() <= closing
    });
    let v_closed = along(u_domain).all(|u| {
        (surface.point_at(u, v_domain.0) - surface.point_at(u, v_domain.1)).norm() <= closing
    });

    let parameters = [
        Parameter {
            domain: u_domain,
            travel: u_travel,
            closed: u_closed,
        },
        Parameter {
            domain: v_domain,
            travel: v_travel,
            closed: v_closed,
        },
    ];
    (size, parameters)
}

/// The distance below which the two ends of geometry of the given `size` are the same place.
fn closing_distance<T: FloatingPoint>(size: T) -> T {
    size * T::default_epsilon().sqrt()
}

/// The length of the polygon through homogeneous control points.
fn polygon_length<'a, T: FloatingPoint, D>(
    control_points: impl Iterator<Item = &'a OPoint<T, D>>,
) -> T
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    control_points
        .filter_map(dehomogenize)
        .tuple_windows()
        .map(|(a, b)| (b - a).norm())
        .fold(T::zero(), |a, b| a + b)
}

/// The state of the solvers of the intersection finders.
pub(crate) type SolverState<T, const N: usize> =
    IterState<SVector<T, N>, SVector<T, N>, (), SMatrix<T, N, N>, (), T>;

/// A problem seen in its [`Scales`].
pub(crate) struct Normalized<'a, O, T, const N: usize> {
    problem: O,
    scales: &'a Scales<T, N>,
}

impl<O, T: FloatingPoint, const N: usize> CostFunction for Normalized<'_, O, T, N>
where
    O: CostFunction<Param = SVector<T, N>, Output = T>,
{
    type Param = SVector<T, N>;
    type Output = T;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let cost = self.problem.cost(&self.scales.denormalize(param))?;
        Ok(cost / (self.scales.size * self.scales.size))
    }
}

impl<O, T: FloatingPoint, const N: usize> Gradient for Normalized<'_, O, T, N>
where
    O: Gradient<Param = SVector<T, N>, Gradient = SVector<T, N>>,
{
    type Param = SVector<T, N>;
    type Gradient = SVector<T, N>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
        let gradient = self.problem.gradient(&self.scales.denormalize(param))?;
        Ok(gradient.component_mul(&self.scales.unit) / (self.scales.size * self.scales.size))
    }
}

/// Choose the intersections among the `candidates` the solver found, one for every pair of leaves.
///
/// A candidate is an intersection when its two points are closer than `minimum_distance`. Several
/// leaves find the same intersection, so the candidates are put in `order`, each run of
/// candidates that are `same` is a group, and the closest of a group is kept.
///
/// `same` is asked about two candidates next to each other in the order, with `false`. When the
/// geometry is `closed` in the parameter they are ordered by, the last group and the first are on
/// the two sides of its seam, and `same` is asked about their ends with `true`.
pub(crate) fn closest_in_groups<I: Clone, T: FloatingPoint>(
    candidates: impl IntoIterator<Item = I>,
    distance: impl Fn(&I) -> T,
    minimum_distance: T,
    order: impl Fn(&I) -> T,
    closed: bool,
    same: impl Fn(&I, &I, bool) -> bool,
) -> Vec<I> {
    let mut groups = candidates
        .into_iter()
        .filter(|candidate| distance(candidate) < minimum_distance)
        .sorted_by(|x, y| order(x).partial_cmp(&order(y)).unwrap_or(Ordering::Equal))
        .map(|candidate| vec![candidate])
        .coalesce(|mut group, next| {
            if same(&group[group.len() - 1], &next[0], false) {
                group.extend(next);
                Ok(group)
            } else {
                Err((group, next))
            }
        })
        .collect_vec();

    if closed && groups.len() > 1 {
        let (first, last) = (&groups[0], &groups[groups.len() - 1]);
        if same(&last[last.len() - 1], &first[0], true) {
            let last = groups.pop().unwrap();
            groups[0].extend(last);
        }
    }

    groups
        .into_iter()
        .filter_map(|group| {
            group.into_iter().min_by(|x, y| {
                distance(x)
                    .partial_cmp(&distance(y))
                    .unwrap_or(Ordering::Equal)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector1;

    use super::*;

    fn scales(size: f64, domain: (f64, f64), closed: bool) -> Scales<f64, 1> {
        let parameter = Parameter {
            domain,
            travel: size,
            closed,
        };
        Scales::new(&[size], [parameter])
    }

    /// The squared distance from a point moving `size` over `domain` to where it is 0.3 of the
    /// way.
    struct Along {
        size: f64,
        domain: (f64, f64),
    }

    impl Along {
        fn offset(&self, t: f64) -> f64 {
            self.size * ((t - self.domain.0) / (self.domain.1 - self.domain.0) - 0.3)
        }
    }

    impl CostFunction for Along {
        type Param = Vector1<f64>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            Ok(self.offset(param[0]).powi(2))
        }
    }

    impl Gradient for Along {
        type Param = Vector1<f64>;
        type Gradient = Vector1<f64>;

        fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
            let speed = self.size / (self.domain.1 - self.domain.0);
            Ok(Vector1::new(2. * self.offset(param[0]) * speed))
        }
    }

    #[test]
    fn the_solver_is_given_the_same_problem_at_any_size_and_over_any_domain() {
        for (size, domain) in [(1., (0., 1.)), (1e-3, (2., 2.5)), (1e3, (-40., 360.))] {
            let scales = scales(size, domain, false);
            let normalized = Normalized {
                problem: Along { size, domain },
                scales: &scales,
            };
            for step in [0., 0.1, 0.3, 0.75, 1.] {
                let param = Vector1::new(step);
                let cost = normalized.cost(&param).unwrap();
                let gradient = normalized.gradient(&param).unwrap();
                assert!((cost - (step - 0.3).powi(2)).abs() < 1e-12);
                assert!((gradient[0] - 2. * (step - 0.3)).abs() < 1e-12);
                // and its parameters are those of the problem
                let there = scales.normalize(&scales.denormalize(&param));
                assert!((there[0] - step).abs() < 1e-12);
            }
            assert!((scales.distance(1e-5) - 1e-5 * size).abs() < 1e-20);
        }
    }

    #[test]
    fn halfway_is_across_the_seam_only_of_closed_geometry() {
        let open = scales(1., (1., 5.), false);
        let closed = scales(1., (1., 5.), true);
        // near each other, either way
        assert_eq!(open.halfway(0, 2., 3.), 2.5);
        assert_eq!(closed.halfway(0, 2., 3.), 2.5);
        // more than half the domain apart
        assert_eq!(open.halfway(0, 1.5, 4.9), 3.2);
        assert!((closed.halfway(0, 1.5, 4.9) - 1.2).abs() < 1e-12);
        assert!((closed.halfway(0, 1.1, 4.5) - 4.8).abs() < 1e-12);
    }

    #[test]
    fn the_closest_of_each_group_of_candidates_is_kept() {
        // a parameter and the distance between the two points there
        let candidates = [
            (0.02, 0.5),
            (0.5, 0.3),
            (0.52, 0.1),
            (0.54, 0.2),
            (0.7, 2.), // not an intersection
            (0.98, 0.4),
        ];
        let choose = |closed: bool| {
            closest_in_groups(
                candidates,
                |candidate| candidate.1,
                1.,
                |candidate| candidate.0,
                closed,
                |x: &(f64, f64), y: &(f64, f64), across| {
                    let apart = (x.0 - y.0).abs();
                    (if across { 1. - apart } else { apart }) < 0.05
                },
            )
        };
        assert_eq!(choose(false), vec![(0.02, 0.5), (0.52, 0.1), (0.98, 0.4)]);
        // the first and the last are on the two sides of the seam
        assert_eq!(choose(true), vec![(0.98, 0.4), (0.52, 0.1)]);
    }
}
