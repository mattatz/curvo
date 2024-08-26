use std::fmt::Display;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, DimName, Point2, Vector2,
};

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{Contains, CurveIntersection, CurveIntersectionSolverOptions},
    region::{CompoundCurve, Region},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperation {
    Union,
    Intersection,
    Difference,
}

impl Display for BooleanOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BooleanOperation::Union => write!(f, "Union"),
            BooleanOperation::Intersection => write!(f, "Intersection"),
            BooleanOperation::Difference => write!(f, "Difference"),
        }
    }
}

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;
    type Option;

    fn union(&self, other: T, option: Self::Option) -> Self::Output;
    fn intersection(&self, other: T, option: Self::Option) -> Self::Output;
    fn difference(&self, other: T, option: Self::Option) -> Self::Output;
    fn boolean(&self, operation: BooleanOperation, other: T, option: Self::Option) -> Self::Output;
}

impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for NurbsCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    // type Output = anyhow::Result<Vec<Region<T>>>;
    type Output = anyhow::Result<(Vec<Region<T>>, Vec<CurveIntersection<Point2<T>, T>>)>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn union(&self, other: &'a NurbsCurve<T, Const<3>>, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Union, other, option)
    }

    fn intersection(
        &self,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        self.boolean(BooleanOperation::Intersection, other, option)
    }

    fn difference(&self, other: &'a NurbsCurve<T, Const<3>>, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Difference, other, option)
    }

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(other, option.clone())?;
        let origin = intersections.clone();

        let near_zero_eps = T::from_f64(1e-2).unwrap();
        let is_point_on_boundary =
            |point: &Point2<T>, normal: &Vector2<T>, a: Point2<T>, b: Point2<T>| -> bool {
                let da = (a - point).normalize();
                let db = (b - point).normalize();
                // check if a point lies on the a & b segment
                // println!("dot: {:?}", da.dot(&db));
                let is_opposite = ComplexField::abs(da.dot(&db) - -T::one()) <= near_zero_eps;
                if is_opposite {
                    return true;
                }
                let a_dot = da.dot(&normal);
                let b_dot = db.dot(&normal);
                a_dot * b_dot > T::zero()
            };

        let a_delta = self.knots_domain_interval() * T::from_f64(1e-1 * 0.5).unwrap();
        let b_delta = other.knots_domain_interval() * T::from_f64(1e-1 * 0.5).unwrap();

        // TODO: filtering intersections at vertex in curves
        let mut intersections = intersections
            .into_iter()
            .filter(|it| {
                let a_pt = &it.a().0;
                let a_tan = self.tangent_at(it.a().1).normalize();
                let a_normal = Vector2::new(-a_tan.y, a_tan.x);

                let a_vertex = is_point_on_boundary(
                    a_pt,
                    &a_normal,
                    other.point_at(it.b().1 - b_delta),
                    other.point_at(it.b().1 + b_delta),
                );

                if !a_vertex {
                    return true;
                }

                let b_pt = &it.b().0;
                let b_tan = other.tangent_at(it.b().1).normalize();
                let b_normal = Vector2::new(-b_tan.y, b_tan.x);

                let b_vertex = is_point_on_boundary(
                    b_pt,
                    &b_normal,
                    self.point_at(it.a().1 - a_delta),
                    self.point_at(it.a().1 + a_delta),
                );
                // println!("a_vertex: {}, b_vertex: {}", a_vertex, b_vertex);

                !b_vertex
            })
            .collect_vec();

        // println!("origin: {}, filtered: {}", origin.len(), intersections.len());

        if intersections.is_empty() {
            anyhow::bail!("Todo: no intersections case");
        }

        intersections.sort_by(|i0, i1| i0.a().1.partial_cmp(&i1.a().1).unwrap());

        // anyhow::ensure!(intersections.len() % 2 == 0, "Odd number of intersections found");

        let mut regions = vec![];

        let start = self.point_at(self.knots_domain().0);
        let other_contains_self_start = other.contains(&start, option.clone())?;
        let mut curves = [self, other].into_iter().enumerate().cycle();

        match operation {
            BooleanOperation::Union => {
                let cycled = intersections
                    .iter()
                    .cycle()
                    .take(intersections.len() + 1)
                    .collect_vec();
                let windows = cycled.windows(2);

                let mut spans = vec![];

                if !other_contains_self_start {
                    curves.next();
                }

                for it in windows {
                    let (i0, i1) = (&it[0], &it[1]);
                    let c = curves.next();
                    if let Some((idx, c)) = c {
                        let params = match idx % 2 {
                            0 => (i0.a().1, i1.a().1),
                            1 => (i0.b().1, i1.b().1),
                            _ => unreachable!(),
                        };
                        if idx % 2 == 0 {
                            let s = try_trim(c, params)?;
                            spans.extend(s);
                        } else {
                            let s = try_trim(c, params)?;
                            spans.extend(s);
                        }
                    }
                }

                regions.push(Region::new(CompoundCurve::from_iter(spans), vec![]));
            }
            BooleanOperation::Intersection => {
                let cycled = intersections
                    .iter()
                    .cycle()
                    .take(intersections.len() + 1)
                    .collect_vec();
                let windows = cycled.windows(2);

                let mut spans = vec![];

                if other_contains_self_start {
                    curves.next();
                }

                for it in windows {
                    let (i0, i1) = (&it[0], &it[1]);
                    let c = curves.next();
                    if let Some((idx, c)) = c {
                        let params = match idx % 2 {
                            0 => (i0.a().1, i1.a().1),
                            1 => (i0.b().1, i1.b().1),
                            _ => unreachable!(),
                        };
                        if idx % 2 == 0 {
                            let s = try_trim(c, params)?;
                            spans.extend(s);
                        } else {
                            let s = try_trim(c, params)?;
                            spans.extend(s);
                        }
                    }
                }

                regions.push(Region::new(CompoundCurve::from_iter(spans), vec![]));
            }
            BooleanOperation::Difference => {
                let skip_count = if other_contains_self_start { 0 } else { 1 };
                let n = intersections.len();
                let cycled = intersections
                    .into_iter()
                    .cycle()
                    .skip(skip_count)
                    .take(n)
                    .collect_vec();
                let chunks = cycled.chunks(2);
                for it in chunks {
                    if it.len() == 2 {
                        let (i0, i1) = (&it[0], &it[1]);
                        let s0 = try_trim(self, (i0.a().1, i1.a().1))?;
                        let s1 = try_trim(other, (i0.b().1, i1.b().1))?;
                        let exterior = [s0, s1].concat();
                        regions.push(Region::new(CompoundCurve::from_iter(exterior), vec![]));
                    }
                }
            }
        }

        Ok((regions, origin))
    }
}

fn try_trim<T: FloatingPoint, D: DimName>(
    curve: &NurbsCurve<T, D>,
    parameters: (T, T),
) -> anyhow::Result<Vec<NurbsCurve<T, D>>>
where
    DefaultAllocator: Allocator<D>,
{
    let (min, max) = (
        parameters.0.min(parameters.1),
        parameters.0.max(parameters.1),
    );
    let inside = if parameters.0 < parameters.1 {
        true
    } else {
        false
    };
    let curves = if inside {
        let (_, tail) = curve.try_trim(min)?;
        let (head, _) = tail.try_trim(max)?;
        vec![head]
    } else {
        let (head, tail) = curve.try_trim(min)?;
        let (_, tail2) = tail.try_trim(max)?;
        vec![tail2, head]
    };

    Ok(curves)
}
