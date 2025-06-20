use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, OPoint,
    OVector, Rotation3, Unit, Vector3, U1,
};

use crate::{curve::NurbsCurve, fillet::segment::Segment, misc::FloatingPoint};

pub type FilletLength<T> = (T, T, T);

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
