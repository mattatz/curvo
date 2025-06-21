use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, OPoint,
    OVector, Rotation3, Unit, Vector3, U1,
};

use crate::{
    curve::NurbsCurve, fillet::segment::Segment, misc::FloatingPoint, region::CompoundCurve,
};

pub type FilletLength<T> = (T, T, T);

/// A trimmed segment with its origin segment
pub struct TrimmedSegment<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    origin: Segment<T, D>,
    trimmed: Segment<T, D>,
}

impl<T: FloatingPoint, D: DimName> TrimmedSegment<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(origin: Segment<T, D>, trimmed: Segment<T, D>) -> Self {
        Self { origin, trimmed }
    }

    pub fn origin(&self) -> &Segment<T, D> {
        &self.origin
    }

    pub fn trimmed(&self) -> &Segment<T, D> {
        &self.trimmed
    }
}

/// A span of a compound curve for fillet operation
pub enum CompoundSegment<S, C> {
    Segment(S),
    Curve(C),
}

/// Decompose a curve into segments
pub fn decompose_into_segments<T: FloatingPoint, D: DimName>(
    curve: &NurbsCurve<T, D>,
) -> Vec<Segment<T, DimNameDiff<D, U1>>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let pts = curve.dehomogenized_control_points();
    pts.windows(2)
        .map(|w| {
            let p0 = &w[0];
            let p1 = &w[1];
            Segment::new(p0.clone(), p1.clone())
        })
        .collect()
}

/// Calculate the fillet length for a given radius and segments
pub fn calculate_fillet_length<T: FloatingPoint, D: DimName, F>(
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

/// Trim a segment by a given length
pub fn trim_segment_by_fillet_length<T: FloatingPoint, D: DimName>(
    segment: &Segment<T, D>,
    prev_fillet: Option<FilletLength<T>>,
    next_fillet: Option<FilletLength<T>>,
) -> Segment<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    let start = match prev_fillet {
        Some((_, length, _)) => segment.trim(length).0.end().clone(),
        None => segment.start().clone(),
    };

    let end = match next_fillet {
        Some((_, length, _)) => segment.trim(segment.length() - length).1.start().clone(),
        None => segment.end().clone(),
    };
    Segment::new(start, end)
}

/// Create a fillet corner curve
pub fn create_fillet_corner_between_trimmed_segments<T: FloatingPoint, D: DimName>(
    segments: &[&TrimmedSegment<T, DimNameDiff<D, U1>>; 2],
    fillet_length: Option<FilletLength<T>>,
) -> anyhow::Result<Option<NurbsCurve<T, D>>>
where
    D: DimName,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let s0 = &segments[0];
    let s1 = &segments[1];
    let t0 = s0.trimmed().tangent();
    let t1 = s1.trimmed().tangent();

    match fillet_length {
        Some((angle, _, radius)) => {
            let corner = s0.origin().end();
            let fillet_arc = create_fillet_arc(
                s0.trimmed().end(),
                corner,
                s1.trimmed().start(),
                t0,
                t1,
                T::pi() - angle,
                radius,
            )?;
            Ok(Some(fillet_arc))
        }
        None => Ok(None),
    }
}

/// Create a fillet arc curve
pub fn create_fillet_arc<T: FloatingPoint, D>(
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
    // <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    // DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
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

/// Connect a list of compound segments into a compound curve
pub fn try_connect_compound_segments<T: FloatingPoint, D: DimName>(
    segments: Vec<Option<CompoundSegment<Segment<T, DimNameDiff<D, U1>>, NurbsCurve<T, D>>>>,
) -> anyhow::Result<CompoundCurve<T, D>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    let mut curves = vec![];
    let mut polyline = vec![];
    for s in segments {
        match s {
            Some(CompoundSegment::Segment(s)) => {
                polyline.push(s.start().clone());
                polyline.push(s.end().clone());
            }
            Some(CompoundSegment::Curve(c)) => {
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
