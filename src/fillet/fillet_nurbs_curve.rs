use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, U1,
};

use crate::{
    curve::NurbsCurve,
    fillet::{
        helper::{calculate_fillet_length, create_fillet_arc, decompose_into_segments, FilletLength},
        segment::Segment,
        Fillet, FilletRadiusOption, FilletRadiusParameterOption,
    },
    misc::FloatingPoint,
    region::CompoundCurve,
};

/// Span of a fillet curve
#[derive(Debug, Clone)]
enum Span<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    Segment(Segment<T, DimNameDiff<D, U1>>),
    Fillet(NurbsCurve<T, D>),
}

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
        let degree = self.degree();

        if degree >= 2 || radius <= T::zero() {
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
    /// let fillet = curve.fillet(FilletRadiusParameterOption::new(0.2, 2.)).unwrap();
    /// assert_eq!(fillet.spans().len(), 3);
    fn fillet(&self, option: FilletRadiusParameterOption<T>) -> Self::Output {
        let degree = self.degree();
        let radius = option.radius();
        let parameter = option.parameter();
        let domain = self.knots_domain();

        anyhow::ensure!(
            domain.0 <= parameter && parameter <= domain.1,
            "Parameter must be in the domain of the curve, but got {}",
            parameter
        );

        if degree >= 2 || radius <= T::zero() {
            return Ok(self.clone().into());
        }

        let segments = decompose_into_segments(self);

        let is_closed = self.is_closed();
        let n = segments.len();
        let m = if is_closed { n + 1 } else { n };

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

        let eps = T::from_f64(1e-6).unwrap();
        let angle_fillet_length = segments
            .iter()
            .cycle()
            .take(m)
            .collect_vec()
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                if i == index {
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

            let start = match prev_fillet {
                Some((_, length, _)) => current.trim(length).0.end().clone(),
                None => current.start().clone(),
            };

            let end = match next_fillet {
                Some((_, length, _)) => current.trim(current.length() - length).1.start().clone(),
                None => current.end().clone(),
            };
            (current, Segment::new(start, end))
        })
        .collect_vec();

    let corners = trimmed
        .iter()
        .cycle()
        .take(m)
        .collect_vec()
        .windows(2)
        .zip(angle_fillet_length.iter())
        .map(|(w, af)| {
            let s0 = w[0];
            let s1 = w[1];
            let t0 = s0.1.tangent();
            let t1 = s1.1.tangent();

            match af {
                Some((angle, _, radius)) => {
                    let corner = s0.0.end();
                    let fillet_arc = create_fillet_arc(
                        s0.1.end(),
                        corner,
                        s1.1.start(),
                        t0,
                        t1,
                        T::pi() - *angle,
                        *radius,
                    )?;
                    Ok(Some(fillet_arc))
                }
                None => Ok(None),
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let spans: Vec<Option<Span<T, D>>> = trimmed
        .into_iter()
        .map(|s| Some(Span::Segment(s.1)))
        .interleave(corners.into_iter().map(|c| c.map(|c| Span::Fillet(c))))
        .collect_vec();

    let mut curves = vec![];
    let mut polyline = vec![];
    for s in spans {
        match s {
            Some(Span::Segment(s)) => {
                polyline.push(s.start().clone());
                polyline.push(s.end().clone());
            }
            Some(Span::Fillet(c)) => {
                polyline.dedup();
                curves.push(NurbsCurve::polyline(&polyline, false));
                curves.push(c);
                polyline.clear();
            }
            None => {}
        }
    }

    if !polyline.is_empty() {
        polyline.dedup();
        curves.push(NurbsCurve::polyline(&polyline, false));
    }

    Ok(CompoundCurve::new_unchecked_aligned(curves))
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
