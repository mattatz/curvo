//! What the intersection finders share: the scales the solver works in, and the choice of the
//! intersections among what it finds.

use std::cmp::Ordering;

use argmin::core::{ArgminFloat, CostFunction, Error, Executor, Gradient, State};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, RealField,
    SMatrix, SVector, U1,
};

use crate::{
    bounding_box::BoundingBox,
    curve::{nurbs_curve::dehomogenize, NurbsCurve},
    misc::FloatingPoint,
    surface::NurbsSurface,
};

use super::{CurveIntersectionSolverOptions, IntersectionBFGS, IntersectionIterState};

/// A parameter of a problem: its domain, how far the geometry travels over it, and the gap between
/// where the geometry is at the start of the domain and where it is at its end.
#[derive(Clone, Copy)]
pub(crate) struct Parameter<T> {
    domain: (T, T),
    travel: T,
    gap: T,
}

/// The scales of a problem with `N` parameters: the size of the geometry and its parameters.
///
/// The solver works on distances divided by the size, and on parameters scaled so that a step of
/// one travels about one size. Its tolerances, and the distance below which two points are an
/// intersection, are then relative: the same geometry at another size, or parameterized over
/// another domain, has the same intersections.
pub(crate) struct Scales<T, const N: usize> {
    size: T,
    /// The distance below which two points are an intersection.
    minimum_distance: T,
    start: SVector<T, N>,
    end: SVector<T, N>,
    /// What a step of one in the solver is in each parameter.
    unit: SVector<T, N>,
    /// Whether the geometry is closed in each parameter: the two ends of the domain the same
    /// place, as far as intersections go.
    closed: [bool; N],
}

impl<T: FloatingPoint, const N: usize> Scales<T, N> {
    /// The scales of geometry of the given `sizes` — the smallest one that is not zero counts, as
    /// an intersection is no larger than the smaller of two objects — with `parameters`, where
    /// two points closer than `minimum_distance`, relative to the size, are an intersection.
    pub(crate) fn new(sizes: &[T], parameters: [Parameter<T>; N], minimum_distance: T) -> Self {
        let size = sizes
            .iter()
            .copied()
            .filter(|size| *size > T::zero())
            .reduce(|a, b| a.min(b))
            .unwrap_or(T::one());
        let minimum_distance = minimum_distance * size;
        Self {
            size,
            minimum_distance,
            start: SVector::from_fn(|i, _| parameters[i].domain.0),
            end: SVector::from_fn(|i, _| parameters[i].domain.1),
            // a domain of no length, or a parameter the geometry does not move with, is left as
            // it is
            unit: SVector::from_fn(|i, _| {
                let Parameter { domain, travel, .. } = parameters[i];
                match (domain.1 > domain.0, travel > T::zero()) {
                    (true, true) => (domain.1 - domain.0) * size / travel,
                    (true, false) => domain.1 - domain.0,
                    (false, _) => T::one(),
                }
            }),
            closed: parameters.map(|parameter| parameter.gap < minimum_distance),
        }
    }

    /// The distance below which two points are an intersection.
    pub(crate) fn minimum_distance(&self) -> T {
        self.minimum_distance
    }

    /// The value of the parameter `index` halfway between `x` and `y`. On geometry closed in
    /// this parameter, values more than half the domain apart are on the two sides of the seam,
    /// and halfway between them is across it.
    pub(crate) fn halfway(&self, index: usize, x: T, y: T) -> T {
        let length = self.end[index] - self.start[index];
        let across = self.closed[index] && (x - y).abs() > length * T::from_f64(0.5).unwrap();
        self.halfway_through(index, x, y, across)
    }

    /// The value of the parameter `index` halfway between `x` and `y`, going `across` the seam of
    /// the geometry or not.
    fn halfway_through(&self, index: usize, x: T, y: T, across: bool) -> T {
        let half = T::from_f64(0.5).unwrap();
        if !across {
            return (x + y) * half;
        }
        let (start, end) = (self.start[index], self.end[index]);
        let halfway = (x + y + (end - start)) * half;
        if halfway > end {
            halfway - (end - start)
        } else {
            halfway
        }
    }

    fn normalize(&self, parameters: &SVector<T, N>) -> SVector<T, N> {
        (parameters - self.start).component_div(&self.unit)
    }

    fn denormalize(&self, normalized: &SVector<T, N>) -> SVector<T, N> {
        normalized.component_mul(&self.unit) + self.start
    }

    /// What the solver finds for the pair of leaves `problem` is about, with the tolerances of
    /// `options`: from the start of the leaves, which are over `leaf` in each parameter, and from
    /// their end.
    ///
    /// From one place only, the solver may go past an intersection of the leaves on its way to
    /// another one, stop short of it where the geometry turns back, or find the first of two
    /// intersections in the leaves and never the second.
    pub(crate) fn solve_leaf<O>(
        &self,
        problem: &O,
        options: &CurveIntersectionSolverOptions<T>,
        leaf: [(T, T); N],
    ) -> [Option<Found<T, N>>; 2]
    where
        T: ArgminFloat,
        O: CostFunction<Param = SVector<T, N>, Output = T>
            + Gradient<Param = SVector<T, N>, Gradient = SVector<T, N>>,
    {
        [
            self.solve(problem, options, SVector::from_fn(|i, _| leaf[i].0)),
            self.solve(problem, options, SVector::from_fn(|i, _| leaf[i].1)),
        ]
    }

    /// Solve `problem` in these scales from the parameters `init`, with the tolerances of
    /// `options`. Returns the parameters the solver stops at, brought back into their domains,
    /// and which of them were past an end.
    ///
    /// An intersection at the end of a domain is found a hair inside it or a hair outside, so
    /// what is outside is clamped rather than refused, and how far apart the geometry is there
    /// decides whether it is an intersection.
    fn solve<O>(
        &self,
        problem: &O,
        options: &CurveIntersectionSolverOptions<T>,
        init: SVector<T, N>,
    ) -> Option<Found<T, N>>
    where
        T: ArgminFloat,
        O: CostFunction<Param = SVector<T, N>, Output = T>
            + Gradient<Param = SVector<T, N>, Gradient = SVector<T, N>>,
    {
        let normalized = Normalized {
            problem,
            scales: self,
        };
        let solver = IntersectionBFGS::new()
            .with_step_size_tolerance(options.step_size_tolerance)
            .with_cost_tolerance(options.cost_tolerance);
        let result = Executor::new(normalized, solver)
            .configure(|state: IntersectionIterState<T, N>| {
                state
                    .param(self.normalize(&init))
                    .inv_hessian(SMatrix::identity())
                    .max_iters(options.max_iters)
            })
            .run()
            .ok()?;
        let found = self.denormalize(result.state().get_best_param()?);
        // a solver that ran away has nothing to say
        if !found.iter().all(|parameter| parameter.is_finite()) {
            return None;
        }
        let clamped =
            SVector::from_fn(|i, _| RealField::clamp(found[i], self.start[i], self.end[i]));
        Some((clamped, std::array::from_fn(|i| clamped[i] != found[i])))
    }

    /// Choose the intersections among the `candidates` the solver found, one for every pair of
    /// leaves: their parameters, `distance_at` which the two objects are that far apart.
    ///
    /// A candidate is an intersection when that distance is below the minimum distance. Two
    /// candidates are one intersection when the geometry is still in contact halfway between
    /// them: that is so of an intersection found from several leaves, and of the candidates
    /// found all along the stretch where two objects touch and stay closer than the minimum
    /// distance. So the candidates are put in the order of the parameter `order`, each run of
    /// candidates that are one intersection is a group, and the closest of a group is kept.
    pub(crate) fn intersections(
        &self,
        candidates: impl IntoIterator<Item = SVector<T, N>>,
        order: usize,
        distance_at: impl Fn(&SVector<T, N>) -> T,
    ) -> Vec<SVector<T, N>> {
        let minimum_distance = self.minimum_distance;
        let candidates = candidates
            .into_iter()
            .map(|parameters| (distance_at(&parameters), parameters))
            .filter(|(distance, _)| *distance < minimum_distance);
        let groups = group_in_order(
            candidates,
            |(_, parameters)| parameters[order],
            self.closed[order],
            |(_, x), (_, y), across| {
                // Next to each other in the order, the two are not on the two sides of its seam:
                // that is asked apart, of the last candidate and the first.
                let halfway = SVector::from_fn(|i, _| {
                    if i == order {
                        self.halfway_through(i, x[i], y[i], across)
                    } else {
                        self.halfway(i, x[i], y[i])
                    }
                });
                distance_at(&halfway) < minimum_distance
            },
        );
        groups
            .into_iter()
            .filter_map(|group| {
                group
                    .into_iter()
                    .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal))
            })
            .map(|(_, parameters)| parameters)
            .collect()
    }
}

/// What the solver found: the parameters, brought back into their domains, and which of them were
/// past an end.
pub(crate) type Found<T, const N: usize> = (SVector<T, N>, [bool; N]);

/// Put `items` in `order` and group each run of items that are the `same`.
///
/// `same` is asked about two items next to each other in the order, with `false`. When the
/// geometry is `closed` in what they are ordered by, the last group and the first are on the two
/// sides of its seam, and `same` is asked about their ends with `true`.
pub(crate) fn group_in_order<I, T: FloatingPoint>(
    items: impl IntoIterator<Item = I>,
    order: impl Fn(&I) -> T,
    closed: bool,
    same: impl Fn(&I, &I, bool) -> bool,
) -> Vec<Vec<I>> {
    let mut groups = items
        .into_iter()
        .sorted_by(|x, y| order(x).partial_cmp(&order(y)).unwrap_or(Ordering::Equal))
        .map(|item| vec![item])
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
}

/// The parameters of two objects when the solver went past the end of one of them and not of the
/// other: what it found is where the two would meet if that one went on, so the candidate is its
/// end and the closest point of the other, by `closest_a` or `closest_b`.
///
/// `a` and `b` are the parameters on each object, brought back into their domains, and whether
/// they were past an end.
pub(crate) fn closest_to_an_end<A, B>(
    a: (A, bool),
    b: (B, bool),
    closest_a: impl FnOnce(&B) -> Option<A>,
    closest_b: impl FnOnce(&A) -> Option<B>,
) -> (A, B) {
    match (a.1, b.1) {
        (true, false) => {
            let closest = closest_b(&a.0);
            (a.0, closest.unwrap_or(b.0))
        }
        (false, true) => {
            let closest = closest_a(&b.0);
            (closest.unwrap_or(a.0), b.0)
        }
        _ => (a.0, b.0),
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
    let parameter = Parameter {
        domain,
        travel: polygon_length(curve.control_points().iter()),
        gap: (curve.point_at(domain.0) - curve.point_at(domain.1)).norm(),
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

    // The gap between the two ends of the domain in a direction is the widest there is along the
    // other direction, which a few places along it are taken to show.
    let (u_domain, v_domain) = surface.knots_domain();
    let along = |(start, end): (T, T)| {
        (0..=4).map(move |i| start + (end - start) * T::from_f64(i as f64 / 4.).unwrap())
    };
    let u_gap = longest(
        &mut along(v_domain)
            .map(|v| (surface.point_at(u_domain.0, v) - surface.point_at(u_domain.1, v)).norm()),
    );
    let v_gap = longest(
        &mut along(u_domain)
            .map(|u| (surface.point_at(u, v_domain.0) - surface.point_at(u, v_domain.1)).norm()),
    );

    let parameters = [
        Parameter {
            domain: u_domain,
            travel: u_travel,
            gap: u_gap,
        },
        Parameter {
            domain: v_domain,
            travel: v_travel,
            gap: v_gap,
        },
    ];
    (size, parameters)
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

/// A problem seen in its [`Scales`].
struct Normalized<'a, O, T, const N: usize> {
    problem: &'a O,
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

#[cfg(test)]
mod tests {
    use nalgebra::Vector1;

    use super::*;

    /// The scales of geometry where two points closer than 0.01 of its size are an intersection.
    fn scales(size: f64, domain: (f64, f64), closed: bool) -> Scales<f64, 1> {
        let parameter = Parameter {
            domain,
            travel: size,
            gap: if closed { 0. } else { size },
        };
        Scales::new(&[size], [parameter], 0.01)
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
                problem: &Along { size, domain },
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
            assert!((scales.minimum_distance() - 0.01 * size).abs() < 1e-20);
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
    fn what_the_solver_finds_past_an_end_is_brought_back_to_it() {
        let options = CurveIntersectionSolverOptions::default();
        // the closest the point gets to its target is 0.3 of the way along a domain that ends
        // before that
        let along = Along {
            size: 2.,
            domain: (0., 4.),
        };
        let init = Vector1::new(0.5);
        let short = scales(2., (0., 1.), false);
        let (found, past) = short.solve(&along, &options, init).unwrap();
        assert_eq!((found[0], past), (1., [true]));

        let whole = scales(2., (0., 4.), false);
        let (found, past) = whole.solve(&along, &options, init).unwrap();
        assert!((found[0] - 1.2).abs() < 1e-6);
        assert_eq!(past, [false]);
    }

    /// The squared distance from a point to a line it crosses at 0.3 and at 0.7, turning back
    /// halfway between them, 0.04 away from it.
    struct ThereAndBack;

    impl ThereAndBack {
        fn distance(t: f64) -> f64 {
            (t - 0.5).powi(2) - 0.04
        }
    }

    impl CostFunction for ThereAndBack {
        type Param = Vector1<f64>;
        type Output = f64;

        fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
            Ok(Self::distance(param[0]).powi(2))
        }
    }

    impl Gradient for ThereAndBack {
        type Param = Vector1<f64>;
        type Gradient = Vector1<f64>;

        fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, Error> {
            let t = param[0];
            Ok(Vector1::new(2. * Self::distance(t) * 2. * (t - 0.5)))
        }
    }

    #[test]
    fn a_leaf_is_solved_from_its_start_and_from_its_end() {
        let options = CurveIntersectionSolverOptions::default();
        let scales = scales(1., (0., 1.), false);
        let solutions = |leaf| {
            let found = scales.solve_leaf(&ThereAndBack, &options, [leaf]);
            found.map(|found| found.map(|(found, _)| (found[0] * 1e6).round() / 1e6))
        };
        // one intersection in the leaf, found from both ends
        assert_eq!(solutions((0.28, 0.32)), [Some(0.3), Some(0.3)]);
        // two intersections in the leaf, each found from the end it is nearer
        assert_eq!(solutions((0.25, 0.75)), [Some(0.3), Some(0.7)]);
        // the solver cannot tell which way to go from where the point turns back, at the start
        // of the leaf: the intersection is found from its end
        assert_eq!(solutions((0.5, 0.8)), [Some(0.5), Some(0.7)]);
    }

    /// A point moving along a line, `distance_at` its parameter from the line it crosses at 0.3
    /// and at 0.7 of a domain of 0 to 1: closer than 0.01 over a stretch around each.
    fn two_crossings(parameters: &Vector1<f64>) -> f64 {
        let t = parameters[0];
        (t - 0.3).abs().min((t - 0.7).abs()) * 0.1
    }

    #[test]
    fn one_intersection_is_kept_for_each_stretch_in_contact() {
        let scales = scales(1., (0., 1.), false);
        let candidates = [0.31, 0.25, 0.5, 0.36, 0.7, 0.72, 0.1].map(Vector1::new);
        let intersections = scales.intersections(candidates, 0, two_crossings);
        assert_eq!(intersections, vec![Vector1::new(0.31), Vector1::new(0.7)]);
    }

    #[test]
    fn candidates_on_the_two_sides_of_a_seam_are_one_intersection() {
        // in contact around the seam, at 0 and at 1, and around 0.5
        let around = |parameters: &Vector1<f64>| {
            let t = parameters[0];
            t.min(1. - t).min((t - 0.5).abs()) * 0.1
        };
        let candidates = [0.02, 0.48, 0.52, 0.97].map(Vector1::new);
        let open = scales(1., (0., 1.), false).intersections(candidates, 0, around);
        assert_eq!(open, [0.02, 0.48, 0.97].map(Vector1::new).to_vec());
        let closed = scales(1., (0., 1.), true).intersections(candidates, 0, around);
        assert_eq!(closed, [0.02, 0.48].map(Vector1::new).to_vec());
    }

    #[test]
    fn past_the_end_of_one_object_the_closest_of_the_other_is_taken() {
        let closest = |a, b| closest_to_an_end(a, b, |_| Some(10), |_| Some("closest"));
        assert_eq!(closest((1, false), ("b", false)), (1, "b"));
        assert_eq!(closest((1, true), ("b", false)), (1, "closest"));
        assert_eq!(closest((1, false), ("b", true)), (10, "b"));
        // both at an end: nothing to look for
        assert_eq!(closest((1, true), ("b", true)), (1, "b"));
    }
}
