use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator};

use crate::{
    boolean::degeneracies::Degeneracy,
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{Contains, CurveIntersectionSolverOptions},
    region::{CompoundCurve, Region},
};

use super::Boolean;

/// Boolean operation for compound curve & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for CompoundCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    type Output = anyhow::Result<Vec<Region<T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: super::operation::BooleanOperation,
        clip: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(clip, option.clone())?;

        let deg = intersections
            .into_iter()
            .map(|it| {
                let deg = Degeneracy::new(&it, it.a().0, it.b().0);
                (it, deg)
            })
            .collect_vec();

        let (regular, deg): (Vec<_>, Vec<_>) = deg
            .into_iter()
            .partition(|(_, deg)| matches!(deg, Degeneracy::None));

        let mut regular = regular.into_iter().map(|(it, _)| it).collect_vec();
        let intersections = if regular.len() % 2 == 0 {
            regular
        } else {
            let max = deg.into_iter().max_by(|x, y| match (x.1, y.1) {
                (Degeneracy::Angle(x), Degeneracy::Angle(y)) => {
                    x.partial_cmp(&y).unwrap_or(Ordering::Equal)
                }
                _ => Ordering::Equal,
            });
            match max {
                Some((max, _)) => {
                    regular.push(max);
                    regular
                }
                _ => regular,
            }
        };

        anyhow::ensure!(
            intersections.len() % 2 == 0,
            "found odd number of intersections: {}",
            intersections.len()
        );

        let indexed = intersections.into_iter().enumerate().collect_vec();

        let head = self.spans().first().unwrap();
        let clip_contains_subject =
            clip.contains(&head.point_at(head.knots_domain().0), option.clone())?;

        todo!()
    }
}
