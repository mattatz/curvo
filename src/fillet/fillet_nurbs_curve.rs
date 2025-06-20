use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, OPoint,
    OVector, Rotation3, Unit, Vector3, U1,
};

use crate::{
    curve::NurbsCurve,
    fillet::{segment::Segment, Fillet, FilletRadiusOption, FilletRadiusParameterOption},
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

        let pts = self.dehomogenized_control_points();

        let segments = pts
            .windows(2)
            .map(|w| {
                let p0 = &w[0];
                let p1 = &w[1];
                Segment::new(p0.clone(), p1.clone())
            })
            .collect_vec();

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

        let pts = self.dehomogenized_control_points();

        let segments = pts
            .windows(2)
            .map(|w| {
                let p0 = &w[0];
                let p1 = &w[1];
                Segment::new(p0.clone(), p1.clone())
            })
            .collect_vec();

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
            .clamp(0, pts.len())
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

type FilletLength<T> = (T, T, T);

/// Calculate the fillet length for a given radius and segments
fn calculate_fillet_length<T: FloatingPoint, D: DimName, F>(
    w: &[&Segment<T, D>; 2],
    radius: T,
    constrain: F,
) -> Option<FilletLength<T>>
where
    DefaultAllocator: Allocator<D>,
    F: Fn(T) -> T,
{
    let half = T::from_f64(0.5).unwrap();

    let s0 = &w[0];
    let s1 = &w[1];
    let t0 = s0.tangent();
    let t1 = s1.tangent();
    let cos_angle = t0.dot(t1).clamp(-T::one(), T::one());
    let angle = cos_angle.acos();
    if angle < T::from_f64(1e-4).unwrap() {
        return None;
    }
    let half_angle = angle * half;

    let tan_half = half_angle.tan();
    let fillet_length = radius * tan_half;

    let actual_fillet_length = constrain(fillet_length);

    let actual_radius = actual_fillet_length / tan_half;
    Some((angle, actual_fillet_length, actual_radius))
}

/// Create a fillet arc curve
fn create_fillet_arc<T: FloatingPoint, D>(
    start: &OPoint<T, DimNameDiff<D, U1>>,
    corner: &OPoint<T, DimNameDiff<D, U1>>,
    _end: &OPoint<T, DimNameDiff<D, U1>>,
    t0: &OVector<T, DimNameDiff<D, U1>>,
    t1: &OVector<T, DimNameDiff<D, U1>>,
    angle: T,
    radius: T,
) -> anyhow::Result<NurbsCurve<T, D>>
where
    D: DimName,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    anyhow::ensure!(
        D::dim() - 1 <= 3,
        "Curve dimension must be less than or equal to 3, but got {}",
        D::dim() - 1
    );

    let theta = angle / T::from_f64(2.0).unwrap();
    let r = radius / theta.sin();

    let bisector = (t1 - t0).normalize();
    let center = corner + bisector * r;

    let x_axis = (start - &center).normalize();

    let t0_3d = to_3d_vector(t0).normalize();
    let t1_3d = to_3d_vector(t1).normalize();
    let rot = rotate_around_axis(&t0_3d, &t1_3d, T::frac_pi_2());

    let dx = to_3d_vector(&x_axis);
    let dy = rot * dx;
    let y_axis = OVector::<T, DimNameDiff<D, U1>>::from_vec(dy.as_slice().to_vec());

    let y_axis = if y_axis.dot(t0) < T::zero() {
        -y_axis
    } else {
        y_axis
    };

    // debug
    // return Ok(NurbsCurve::polyline( &vec![start.clone(), center.clone(), end.clone()], false,));
    // return NurbsCurve::try_circle(&center, &x_axis, &y_axis, radius);
    NurbsCurve::try_arc(
        &center,
        &x_axis,
        &y_axis,
        radius,
        T::zero(),
        T::pi() - angle.abs(),
    )
}

/// Convert a 2d dimension vector to 3D vector
/// if D is 2, return the vector with z = 0
/// if D is 3, return the vector itself
fn to_3d_vector<T: FloatingPoint, D: DimName>(v: &OVector<T, D>) -> Vector3<T>
where
    DefaultAllocator: Allocator<D>,
{
    if D::dim() == 2 {
        Vector3::new(v[0], v[1], T::zero())
    } else {
        let v = v.as_slice().to_vec();
        Vector3::new(v[0], v[1], v[2])
    }
}

/// Rotate a vector around an axis
fn rotate_around_axis<T: FloatingPoint>(
    dx: &Vector3<T>,
    dy: &Vector3<T>,
    theta: T,
) -> Rotation3<T> {
    let n = dx.cross(dy).normalize();
    let axis = Unit::new_normalize(n);
    Rotation3::from_axis_angle(&axis, theta)
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
