use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff,
    DimNameSub, OPoint, OVector, Point2, Vector2, U1,
};

use crate::{curve::NurbsCurve, fillet::Fillet, misc::FloatingPoint, region::CompoundCurve};

#[derive(Debug, Clone, Copy)]
pub struct FilletRadiusOption<T: FloatingPoint> {
    radius: T,
}

impl<T: FloatingPoint> FilletRadiusOption<T> {
    pub fn new(radius: T) -> Self {
        Self { radius }
    }

    pub fn radius(&self) -> T {
        self.radius
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FilletRadiusParameterOption<T: FloatingPoint> {
    radius: T,
    parameter: T,
}

impl<T: FloatingPoint> FilletRadiusParameterOption<T> {
    pub fn new(radius: T, parameter: T) -> Self {
        Self { radius, parameter }
    }

    pub fn radius(&self) -> T {
        self.radius
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FilletDistanceOption<T: FloatingPoint> {
    distance: T,
}

impl<T: FloatingPoint> FilletDistanceOption<T> {
    pub fn new(distance: T) -> Self {
        Self { distance }
    }

    pub fn distance(&self) -> T {
        self.distance
    }
}

/// Segment of a curve
#[derive(Debug, Clone)]
struct Segment<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    start: OPoint<T, D>,
    end: OPoint<T, D>,
    tangent: OVector<T, D>,
    length: T,
}

impl<T: FloatingPoint, D: DimName> Segment<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn new(start: OPoint<T, D>, end: OPoint<T, D>) -> Self {
        let tangent = &end - &start;
        let length = tangent.norm();
        Self {
            start,
            end,
            tangent: tangent.normalize(),
            length,
        }
    }

    fn start(&self) -> &OPoint<T, D> {
        &self.start
    }

    fn end(&self) -> &OPoint<T, D> {
        &self.end
    }

    fn tangent(&self) -> &OVector<T, D> {
        &self.tangent
    }

    fn length(&self) -> T {
        self.length
    }

    /// Trim the segment by a given length
    fn trim(&self, length: T) -> (Self, Self) {
        let t = self.tangent() * length;
        let end = self.start() + &t;
        (
            Self::new(self.start().clone(), end.clone()),
            Self::new(end, self.end().clone()),
        )
    }
}

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

        let half = T::from_f64(0.5).unwrap();

        // calculate angle and fillet length for each segment
        let angle_fillet_length = segments
            .iter()
            .cycle()
            .take(m)
            .collect_vec()
            .windows(2)
            .enumerate()
            .map(|(_i, w)| {
                let s0 = &w[0];
                let s1 = &w[1];
                let t0 = s0.tangent();
                let t1 = s1.tangent();
                let cos_angle = t0.dot(&t1).clamp(-T::one(), T::one());
                let angle = cos_angle.acos();
                if angle < T::from_f64(1e-4).unwrap() {
                    return None;
                }
                let half_angle = angle * half;

                // let sin_half = half_angle.sin();
                // let cos_half = half_angle.cos();
                // let fillet_length = radius * cos_half / sin_half;

                let tan_half = half_angle.tan();
                let fillet_length = radius * tan_half;

                let l0 = s0.length() * half;
                let l1 = s1.length() * half;
                Some((angle, fillet_length.min(l0).min(l1)))
            })
            .collect_vec();

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
                    Some((_, length)) => current.trim(length).0.end().clone(),
                    None => current.start().clone(),
                };

                let end = match next_fillet {
                    Some((_, length)) => current.trim(current.length() - length).1.start().clone(),
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
                    Some((angle, _)) => {
                        let corner = s0.0.end();
                        let fillet_arc = create_fillet_arc(
                            s0.1.end(),
                            corner,
                            s1.1.start(),
                            t0,
                            t1,
                            T::pi() - *angle,
                            radius,
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
}

/// Create a fillet arc curve
fn create_fillet_arc<T: FloatingPoint, D: DimName>(
    start: &OPoint<T, DimNameDiff<D, U1>>,
    corner: &OPoint<T, DimNameDiff<D, U1>>,
    end: &OPoint<T, DimNameDiff<D, U1>>,
    t0: &OVector<T, DimNameDiff<D, U1>>,
    t1: &OVector<T, DimNameDiff<D, U1>>,
    angle: T,
    radius: T,
) -> anyhow::Result<NurbsCurve<T, D>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    let theta = angle / T::from_f64(2.0).unwrap();
    let r = radius / theta.sin();

    // let bisector = ((t1 + t0) / T::from_f64(2.0).unwrap() - t0).normalize();
    let bisector = (t1 - t0).normalize();
    let center = corner + bisector * r;

    let x_axis = (start - &center).normalize();
    let y_axis = if D::dim() > 3 {
        let n = t0.cross(t1);
        n
    } else {
        let x = x_axis.as_slice();
        OVector::<T, DimNameDiff<D, U1>>::from_vec(vec![x[1], -x[0]])
    };
    let y_axis = if y_axis.dot(&t0) < T::zero() {
        -y_axis
    } else {
        y_axis
    };

    // debug
    // return Ok(NurbsCurve::polyline( &vec![start.clone(), center.clone(), end.clone()], false,));
    // return NurbsCurve::try_circle(&center, &x_axis, &y_axis, radius);
    NurbsCurve::try_arc(&center, &x_axis, &y_axis, radius, T::zero(), T::pi() - angle.abs())
}
