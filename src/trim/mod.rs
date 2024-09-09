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

        let inside = parameters.0 < parameters.1;
        let range = min..=max;
        let eps = T::default_epsilon();
        let curves = self
            .spans()
            .iter()
            .map(|curve| {
                let (d0, d1) = curve.knots_domain();

                // use epsilon to avoid to trim at the terminal (start / end) of the curve
                let curve_domain = (d0 + eps)..=(d1 - eps);

                match (curve_domain.contains(&min), curve_domain.contains(&max)) {
                    (true, true) => {
                        if inside {
                            curve.try_trim_range((min, max))
                        } else {
                            let (head, tail) = curve.try_trim(min)?;
                            let (_, tail2) = tail.try_trim(max)?;
                            Ok(vec![head, tail2])
                        }
                    }
                    (true, false) => {
                        curve.try_trim(min).map(
                            |(head, tail)| {
                                if inside {
                                    vec![tail]
                                } else {
                                    vec![head]
                                }
                            },
                        )
                    }
                    (false, true) => {
                        curve.try_trim(max).map(
                            |(head, tail)| {
                                if inside {
                                    vec![head]
                                } else {
                                    vec![tail]
                                }
                            },
                        )
                    }
                    (false, false) => {
                        let contains = range.contains(&d0);
                        if inside {
                            if contains {
                                Ok(vec![curve.clone()])
                            } else {
                                Ok(vec![])
                            }
                        } else {
                            if contains {
                                Ok(vec![])
                            } else {
                                Ok(vec![curve.clone()])
                            }
                        }
                    }
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect_vec();

        Ok(curves)
    }
}
