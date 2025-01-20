use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};
use rand::{rngs::ThreadRng, Rng};

use crate::{
    curve::NurbsCurve,
    misc::{three_points_are_flat, FloatingPoint},
};

use super::Tessellation;

impl<T: FloatingPoint, D: DimName> Tessellation for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Option = Option<T>;
    type Output = Vec<OPoint<T, DimNameDiff<D, U1>>>;
    /// Tessellate the curve using an adaptive algorithm
    /// this `adaptive` means that the curve will be tessellated based on the curvature of the curve
    fn tessellate(&self, tolerance: Self::Option) -> Self::Output {
        if self.degree() == 1 {
            return self.dehomogenized_control_points();
        }

        let mut rng = rand::thread_rng();
        let tol = tolerance.unwrap_or(T::from_f64(1e-3).unwrap());
        let (start, end) = self.knots_domain();
        tessellate_adaptive(self, start, end, tol, &mut rng)
    }
}

/// Tessellate the curve using an adaptive algorithm recursively
/// if the curve between [start ~ end] is flat enough, it will return the two end points
fn tessellate_adaptive<T: FloatingPoint, D>(
    curve: &NurbsCurve<T, D>,
    start: T,
    end: T,
    tol: T,
    rng: &mut ThreadRng,
) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
where
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let p1 = curve.point_at(start);
    let delta = end - start;
    if delta < T::from_f64(1e-8).unwrap() {
        return vec![p1];
    }

    let p3 = curve.point_at(end);

    let t = 0.5_f64 + 0.2_f64 * rng.gen::<f64>();
    let mid = start + delta * T::from_f64(t).unwrap();
    let p2 = curve.point_at(mid);

    let diff = &p1 - &p3;
    let diff2 = &p1 - &p2;
    if (diff.dot(&diff) < tol && diff2.dot(&diff2) > tol)
        || !three_points_are_flat(&p1, &p2, &p3, tol)
    {
        let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
        let mut left_pts = tessellate_adaptive(curve, start, exact_mid, tol, rng);
        let right_pts = tessellate_adaptive(curve, exact_mid, end, tol, rng);
        left_pts.pop();
        [left_pts, right_pts].concat()
    } else {
        vec![p1, p3]
    }
}
