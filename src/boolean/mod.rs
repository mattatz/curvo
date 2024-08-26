use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimName};

use crate::{
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{Contains, CurveIntersectionSolverOptions},
    region::{CompoundCurve, Region},
};

pub enum BooleanOperation {
    Union,
    Intersection,
    Difference,
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
    type Output = anyhow::Result<Vec<Region<T>>>;
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
        let mut intersections = self.find_intersections(other, option.clone())?;
        if intersections.is_empty() {
            todo!("no intersections case");
        }

        intersections.sort_by(|i0, i1| i0.a().1.partial_cmp(&i1.a().1).unwrap());

        anyhow::ensure!(
            intersections.len() % 2 == 0,
            "Odd number of intersections found"
        );

        let self_mid_parameter =
            (intersections[0].a().1 + intersections[1].a().1) / T::from_f64(2.).unwrap();
        let other_mid_parameter =
            (intersections[0].b().1 + intersections[1].b().1) / T::from_f64(2.).unwrap();

        let other_contains_self_mid =
            other.contains(&self.point_at(self_mid_parameter), option.clone())?;
        let self_contains_other_mid =
            self.contains(&other.point_at(other_mid_parameter), option.clone())?;

        // println!("other_contains_self_mid: {:?}, self_contains_other_mid: {:?}", other_contains_self_mid, self_contains_other_mid, );

        let mut regions = vec![];

        match operation {
            BooleanOperation::Union => {
                let cycled = intersections
                    .iter()
                    .cycle()
                    .take(intersections.len() + 1)
                    .collect_vec();
                let windows = cycled.windows(2);
                let mut spans = vec![];
                for (i, it) in windows.enumerate() {
                    let (i0, i1) = (&it[0], &it[1]);
                    let s = if i % 2 == 0 {
                        try_trim(self, (i0.a().1, i1.a().1), !other_contains_self_mid)?
                    } else {
                        try_trim(other, (i0.b().1, i1.b().1), !self_contains_other_mid)?
                    };
                    spans.extend(s);
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
                for (i, it) in windows.enumerate() {
                    let (i0, i1) = (&it[0], &it[1]);
                    let s = if i % 2 == 0 {
                        try_trim(other, (i0.b().1, i1.b().1), self_contains_other_mid)?
                    } else {
                        try_trim(self, (i0.a().1, i1.a().1), other_contains_self_mid)?
                    };
                    spans.extend(s);
                }
                regions.push(Region::new(CompoundCurve::from_iter(spans), vec![]));
            }
            BooleanOperation::Difference => {
                let chunks = intersections.chunks(2);
                for it in chunks {
                    let (i0, i1) = (&it[0], &it[1]);
                    let s0 = try_trim(self, (i0.a().1, i1.a().1), !other_contains_self_mid)?;
                    let s1 = try_trim(other, (i0.b().1, i1.b().1), self_contains_other_mid)?;
                    let exterior = [s0, s1].concat();
                    regions.push(Region::new(CompoundCurve::from_iter(exterior), vec![]));
                }
            }
        }

        Ok(regions)
    }
}

fn try_trim<T: FloatingPoint, D: DimName>(
    curve: &NurbsCurve<T, D>,
    parameters: (T, T),
    inside: bool,
) -> anyhow::Result<Vec<NurbsCurve<T, D>>>
where
    DefaultAllocator: Allocator<D>,
{
    let (min, max) = (
        parameters.0.min(parameters.1),
        parameters.0.max(parameters.1),
    );
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
