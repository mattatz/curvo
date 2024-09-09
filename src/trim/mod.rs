use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{curve::NurbsCurve, misc::FloatingPoint, region::CompoundCurve};

/// Trim curve by range parameters.
pub trait TrimRange<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>>;
}

impl<T: FloatingPoint, D: DimName> TrimRange<T, D> for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>> {
        let (min, max) = (
            parameters.0.min(parameters.1),
            parameters.0.max(parameters.1),
        );
        let inside = parameters.0 < parameters.1;
        let curves = if inside {
            let (_, tail) = self.try_trim(min)?;
            let (head, _) = tail.try_trim(max)?;
            vec![head]
        } else {
            let (head, tail) = self.try_trim(min)?;
            let (_, tail2) = tail.try_trim(max)?;
            vec![tail2, head]
        };

        Ok(curves)
    }
}

impl<T: FloatingPoint, D: DimName> TrimRange<T, D> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn try_trim_range(&self, parameters: (T, T)) -> anyhow::Result<Vec<NurbsCurve<T, D>>> {
        let (min, max) = (
            parameters.0.min(parameters.1),
            parameters.0.max(parameters.1),
        );
        let start = self
            .spans()
            .iter()
            .find_position(|span| {
                let (d0, d1) = span.knots_domain();
                (d0..=d1).contains(&min)
            })
            .ok_or(anyhow::anyhow!("Failed to find start span"))?;
        let end = self
            .spans()
            .iter()
            .find_position(|span| {
                let (d0, d1) = span.knots_domain();
                (d0..=d1).contains(&max)
            })
            .ok_or(anyhow::anyhow!("Failed to find end span"))?;

        let inside = parameters.0 < parameters.1;
        let range = min..=max;
        let curves = if inside {
            (start.0..=end.0)
                .map(|i| {
                    let curve = &self.spans()[i];
                    let (d0, d1) = curve.knots_domain();
                    let curve_domain = d0..=d1;
                    match (curve_domain.contains(&min), curve_domain.contains(&max)) {
                        (true, true) => curve.try_trim_range((min, max)),
                        (true, false) => curve.try_trim(min).map(|(_, tail)| vec![tail]),
                        (false, true) => curve.try_trim(max).map(|(head, _)| vec![head]),
                        (false, false) => {
                            if range.contains(&d0) {
                                Ok(vec![curve.clone()])
                            } else {
                                Ok(vec![])
                            }
                        }
                    }
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect_vec()
        } else {
            (start.0..=end.0)
                .map(|i| {
                    let curve = &self.spans()[i];
                    let (d0, d1) = curve.knots_domain();
                    let curve_domain = d0..=d1;
                    match (curve_domain.contains(&min), curve_domain.contains(&max)) {
                        (true, true) => {
                            let (head, tail) = curve.try_trim(min)?;
                            let (_, tail2) = tail.try_trim(max)?;
                            Ok(vec![head, tail2])
                        }
                        (true, false) => curve.try_trim(min).map(|(head, _)| vec![head]),
                        (false, true) => curve.try_trim(max).map(|(_, tail)| vec![tail]),
                        (false, false) => {
                            if range.contains(&d0) {
                                Ok(vec![])
                            } else {
                                Ok(vec![curve.clone()])
                            }
                        }
                    }
                })
                .collect::<anyhow::Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect_vec()
        };
        Ok(curves)
    }
}
