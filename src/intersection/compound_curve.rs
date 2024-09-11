use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};
use num_traits::Float;

use crate::{curve::NurbsCurve, misc::FloatingPoint, region::CompoundCurve};

use super::{
    CompoundCurveIntersection, CurveIntersectionSolverOptions, HasIntersection, Intersection,
};

impl<'a, T, D> Intersection<'a, &'a NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<CompoundCurveIntersection<'a, T, D>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points with another curve
    #[allow(clippy::type_complexity)]
    fn find_intersections(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        options: Self::Option,
    ) -> Self::Output {
        let res: anyhow::Result<Vec<_>> = self
            .spans()
            .iter()
            .map(|span| {
                span.find_intersections(other, options.clone())
                    .map(|intersections| {
                        intersections
                            .into_iter()
                            .map(|it| CompoundCurveIntersection::new(span, other, it))
                            .collect_vec()
                    })
            })
            .collect();

        let mut res = res?;
        let eps = T::from_f64(1e-2).unwrap();

        (0..res.len()).circular_tuple_windows().for_each(|(a, b)| {
            if a != b {
                let ia = res[a].last();
                let ib = res[b].first();
                let cull = match (ia, ib) {
                    (Some(ia), Some(ib)) => {
                        // cull the last point in res[a] if it is too close to the first point in res[b]
                        let da = Float::abs(ia.a().0.knots_domain().1 - ia.a().2);
                        let db = Float::abs(ib.a().0.knots_domain().0 - ib.a().2);
                        da < eps && db < eps
                    }
                    _ => false,
                };
                if cull {
                    res[a].pop();
                }
            }
        });

        Ok(res.into_iter().flatten().collect())
    }
}
