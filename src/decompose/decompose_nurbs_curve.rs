use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{curve::NurbsCurve, knot::KnotVector, misc::FloatingPoint, prelude::Decompose};

impl<T: FloatingPoint, D: DimName> Decompose for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<NurbsCurve<T, D>>;

    /// Decompose the curve into a set of Bezier segments of the same degree
    fn try_decompose(&self) -> anyhow::Result<Self::Output> {
        let mut cloned = self.clone();
        if !cloned.is_clamped() {
            cloned.try_clamp()?;
        }

        let knot_mults = cloned.knots().multiplicity();
        let req_mult = cloned.degree() + 1;

        for knot_mult in knot_mults.iter() {
            if knot_mult.multiplicity() < req_mult {
                let knots_insert = vec![*knot_mult.knot(); req_mult - knot_mult.multiplicity()];
                cloned.try_refine_knot(knots_insert)?;
            }
        }

        let div = cloned.knots().len() / req_mult - 1;
        if div <= 1 {
            Ok(vec![cloned])
        } else {
            let knot_length = req_mult * 2;
            let segments = (0..div)
                .map(|i| {
                    let start = i * req_mult;
                    let end = start + knot_length;
                    let knots = cloned.knots().as_slice()[start..end].to_vec();
                    let control_points =
                        cloned.control_points()[start..(start + req_mult)].to_vec();
                    NurbsCurve::new_unchecked(
                        cloned.degree(),
                        control_points,
                        KnotVector::new(knots),
                    )
                })
                .collect_vec();
            Ok(segments)
        }
    }
}
