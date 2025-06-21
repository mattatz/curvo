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

type CompoundSegmentPreprocess<T, D> =
    CompoundSegment<Segment<T, DimNameDiff<D, U1>>, NurbsCurve<T, D>>;

impl<T: FloatingPoint, D: DimName> Fillet<FilletRadiusOption<T>> for CompoundCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Output = anyhow::Result<CompoundCurve<T, D>>;

    /// Fillet the sharp corners of the curve with a given radius
    fn fillet(&self, option: FilletRadiusOption<T>) -> Self::Output {
        let radius = option.radius();

        let segments = decompose_into_compound_segments(self);

        let is_closed = self.is_closed(None);
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
            .map(|w| match (w[0], w[1]) {
                (CompoundSegment::Segment(s0), CompoundSegment::Segment(s1)) => {
                    calculate_fillet_length(&[s0, s1], radius, |length| {
                        let min = s0.length().min(s1.length());
                        let l = min * half - eps;
                        length.min(l)
                    })
                }
                _ => None,
            })
            .collect_vec();

        fillet_compound_curve(segments, angle_fillet_length, is_closed)
    }
}

impl<T: FloatingPoint, D: DimName> Fillet<FilletRadiusParameterOption<T>> for CompoundCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Output = anyhow::Result<CompoundCurve<T, D>>;

    /// Only fillet the sharp corner at the specified parameter position with a given radius
    fn fillet(&self, option: FilletRadiusParameterOption<T>) -> Self::Output {
        let parameter = option.parameter();
        let domain = self.knots_domain();

        anyhow::ensure!(
            domain.0 <= parameter && parameter <= domain.1,
            "Parameter must be in the domain of the curve, but got {}",
            parameter
        );

        let segments = decompose_into_compound_segments(self);

        let is_closed = self.is_closed(None);
        let n = segments.len();
        let m = if is_closed { n + 1 } else { n };

        todo!()
    }
}

/// Decompose a compound curve into a list of segments and curves
fn decompose_into_compound_segments<T: FloatingPoint, D: DimName>(
    curve: &CompoundCurve<T, D>,
) -> Vec<CompoundSegmentPreprocess<T, D>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    curve
        .spans()
        .iter()
        .flat_map(|c| {
            if c.degree() > 1 {
                vec![CompoundSegment::Curve(c.clone())]
            } else {
                let segments = decompose_into_segments(c);
                segments
                    .into_iter()
                    .map(|s| CompoundSegment::Segment(s))
                    .collect_vec()
            }
        })
        .collect_vec()
}

/// Fillet the sharp corners of the curve with a given radius
fn fillet_compound_curve<T: FloatingPoint, D>(
    segments: Vec<CompoundSegmentPreprocess<T, D>>,
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
        .map(|(i, current)| match current {
            CompoundSegment::Segment(current) => {
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
                let s = TrimmedSegment::new(current, trimmed);
                CompoundSegment::Segment(s)
            }
            CompoundSegment::Curve(c) => CompoundSegment::Curve(c.clone()),
        })
        .collect_vec();

    let corners = trimmed
        .iter()
        .cycle()
        .take(m)
        .collect_vec()
        .windows(2)
        .zip(angle_fillet_length.into_iter())
        .map(|(w, af)| {
            let s0 = &w[0];
            let s1 = &w[1];
            match (s0, s1) {
                (CompoundSegment::Segment(s0), CompoundSegment::Segment(s1)) => {
                    create_fillet_corner_between_trimmed_segments(&[s0, s1], af)
                }
                _ => Ok(None),
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let spans = trimmed
        .into_iter()
        .map(|s| match s {
            CompoundSegment::Segment(s) => Some(CompoundSegment::Segment(s.trimmed().clone())),
            CompoundSegment::Curve(c) => Some(CompoundSegment::Curve(c)),
        })
        .interleave(
            corners
                .into_iter()
                .map(|c| c.map(|c| CompoundSegment::Curve(c))),
        )
        .collect_vec();

    try_connect_compound_segments(spans)
}
