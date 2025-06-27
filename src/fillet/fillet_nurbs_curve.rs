use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, U1,
};

use crate::{
    curve::NurbsCurve,
    fillet::{
        helper::{
            calculate_fillet_length, create_fillet_corner_between_trimmed_segments,
            decompose_into_segments, trim_segment_by_fillet_length, try_connect_compound_segments,
            CompoundSegment, FilletLength, TrimmedSegment,
        },
        segment::Segment,
        Fillet, FilletRadiusOption, FilletRadiusParameterOption,
    },
    misc::FloatingPoint,
    region::CompoundCurve,
};

impl<T: FloatingPoint, D: DimName> Fillet<FilletRadiusOption<T>> for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Output = anyhow::Result<CompoundCurve<T, D>>;

    /// Fillet the sharp corners of the curve with a given radius
    /// # Example
    /// ```
    /// use nalgebra::Point2;
    /// use curvo::prelude::*;
    ///
    /// let square = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(-1.0, 1.0),
    ///     Point2::new(-1.0, -1.0),
    /// ];
    /// let curve = NurbsCurve2D::polyline(&square, false);
    /// let fillet = curve.fillet(FilletRadiusOption::new(0.2)).unwrap();
    /// assert_eq!(fillet.spans().len(), 8);
    /// ```
    fn fillet(&self, option: FilletRadiusOption<T>) -> Self::Output {
        let radius = option.radius();
        anyhow::ensure!(
            radius > T::zero(),
            "Radius must be positive, but got {}",
            radius
        );

        let degree = self.degree();
        if degree >= 2 {
            return Ok(self.clone().into());
        }

        let segments = decompose_into_segments(self);

        let is_closed = self.is_closed();
        let n = segments.len();
        let m = if is_closed { n + 1 } else { n };

        // calculate angle and fillet length for each segment
        let half = T::from_f64(0.5).unwrap();
        let eps = T::from_f64(1e-6).unwrap();
        let angle_fillet_length = segments
            .iter()
            .cycle()
            .take(m)
            .collect_vec()
            .windows(2)
            .map(|w| {
                calculate_fillet_length(&[w[0], w[1]], radius, |length| {
                    let min = w[0].length().min(w[1].length());
                    let l = min * half - eps;
                    length.min(l)
                })
            })
            .collect_vec();

        fillet_curve(segments, angle_fillet_length, is_closed)
    }
}

impl<T: FloatingPoint, D: DimName> Fillet<FilletRadiusParameterOption<T>> for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Output = anyhow::Result<CompoundCurve<T, D>>;

    /// Only fillet the sharp corner at the specified parameter position with a given radius
    /// # Example
    /// ```
    /// use nalgebra::Point2;
    /// use curvo::prelude::*;
    ///
    /// let square = vec![
    ///     Point2::new(-1.0, -1.0),
    ///     Point2::new(1.0, -1.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(-1.0, 1.0),
    ///     Point2::new(-1.0, -1.0),
    /// ];
    /// let curve = NurbsCurve2D::polyline(&square, false);
    /// let fillet = curve.fillet(FilletRadiusParameterOption::new(0.2, vec![2.])).unwrap();
    /// assert_eq!(fillet.spans().len(), 3);
    fn fillet(&self, option: FilletRadiusParameterOption<T>) -> Self::Output {
        let radius = option.radius();
        anyhow::ensure!(
            radius > T::zero(),
            "Radius must be positive, but got {}",
            radius
        );

        let degree = self.degree();
        let parameters = option.parameters();
        let domain = self.knots_domain();

        // Validate all parameters are within domain
        for &parameter in parameters {
            anyhow::ensure!(
                domain.0 <= parameter && parameter <= domain.1,
                "Parameter must be in the domain of the curve, but got {}",
                parameter
            );
        }

        if degree >= 2 {
            return Ok(self.clone().into());
        }

        let segments = decompose_into_segments(self);

        let is_closed = self.is_closed();
        let n = segments.len();
        let m = if is_closed { n + 1 } else { n };

        // Calculate indices for all parameters
        let indices = parameters
            .iter()
            .map(|&parameter| {
                let index = self
                    .knots()
                    .floor(parameter)
                    .ok_or(anyhow::anyhow!(
                        "Parameter must be in the domain of the curve, but got {}",
                        parameter
                    ))?
                    .clamp(0, segments.len() + 1)
                    - 1;

                if !is_closed && index >= n - 1 {
                    anyhow::bail!("Parameter too large, got {}", parameter);
                }
                Ok(index)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let eps = T::from_f64(1e-6).unwrap();
        let angle_fillet_length = segments
            .iter()
            .cycle()
            .take(m)
            .collect_vec()
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                if indices.contains(&i) {
                    calculate_fillet_length(&[w[0], w[1]], radius, |length| {
                        let min = w[0].length().min(w[1].length());
                        let l = min - eps;
                        length.min(l)
                    })
                } else {
                    None
                }
            })
            .collect_vec();

        fillet_curve(segments, angle_fillet_length, is_closed)
    }
}

/// Fillet the sharp corners of the curve with a given radius
fn fillet_curve<T: FloatingPoint, D>(
    segments: Vec<Segment<T, DimNameDiff<D, U1>>>,
    angle_fillet_length: Vec<Option<FilletLength<T>>>,
    is_closed: bool,
) -> anyhow::Result<CompoundCurve<T, D>>
where
    D: DimName,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    let n = segments.len();
    let m = if is_closed { n + 1 } else { n };
    let l = angle_fillet_length.len();

    let trimmed = segments
        .into_iter()
        .enumerate()
        .map(|(i, current)| {
            let prev_fillet = if i != 0 || is_closed {
                angle_fillet_length[(i + l - 1) % l]
            } else {
                None
            };
            let next_fillet = if i != n - 1 || is_closed {
                angle_fillet_length[i % l]
            } else {
                None
            };
            let trimmed = trim_segment_by_fillet_length(&current, prev_fillet, next_fillet);
            TrimmedSegment::new(current, trimmed)
        })
        .collect_vec();

    let corners = trimmed
        .iter()
        .cycle()
        .take(m)
        .collect_vec()
        .windows(2)
        .zip(angle_fillet_length.into_iter())
        .map(|(w, af)| create_fillet_corner_between_trimmed_segments(&[w[0], w[1]], af))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let spans = trimmed
        .into_iter()
        .map(|s| Some(CompoundSegment::Segment(s.trimmed().clone())))
        .interleave(corners.into_iter().map(|c| c.map(CompoundSegment::Curve)))
        .collect_vec();

    try_connect_compound_segments(spans)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Point2;

    use crate::curve::NurbsCurve2D;

    use super::*;

    /// Test if the fillet curve is connected
    #[test]
    fn test_fillet_curve_connectivity() {
        let points = vec![
            Point2::new(-0.5, 2.),
            Point2::new(0.5, 2.),
            Point2::new(0.5, 1.),
            Point2::new(1.5, 1.),
            Point2::new(1.5, -1.),
            Point2::new(-1.5, -1.),
            Point2::new(-1.5, 1.),
            Point2::new(-0.5, 1.),
            Point2::new(-0.5, 2.),
        ]
        .into_iter()
        .collect_vec();
        let curve = NurbsCurve2D::polyline(&points, false);
        let fillet = curve.fillet(FilletRadiusOption::new(0.2)).unwrap();
        let spans = fillet.spans();
        spans.windows(2).for_each(|w| {
            let s0 = &w[0];
            let s1 = &w[1];
            let end = s0.point_at(s0.knots_domain().1);
            let start = s1.point_at(s1.knots_domain().0);
            assert_relative_eq!(end, start, epsilon = 1e-6);
        });
    }
}
