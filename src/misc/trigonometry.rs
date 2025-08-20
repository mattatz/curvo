use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, RealField};

pub fn three_points_are_flat<T: RealField + Copy, D: DimName>(
    p1: &OPoint<T, D>,
    p2: &OPoint<T, D>,
    p3: &OPoint<T, D>,
    tolerance: T,
) -> bool
where
    DefaultAllocator: Allocator<D>,
{
    let p21 = p2 - p1;
    let p31 = p3 - p1;
    if D::dim() == 2 {
        (p21[0] * p31[1] - p21[1] * p31[0]).abs() < tolerance
    } else {
        let norm = p21.cross(&p31);
        let area = norm.dot(&norm);
        area < tolerance
    }
}

/// Find the closest point on a segment
/// * `pt` - point to project
/// * `start` - start point of segment
/// * `end` - end point of segment
/// * `u0` - first param of segment
/// * `u1` - second param of segment
pub fn segment_closest_point<T: RealField + Copy, D: DimName>(
    pt: &OPoint<T, D>,
    start: &OPoint<T, D>,
    end: &OPoint<T, D>,
    u0: T,
    u1: T,
) -> (T, OPoint<T, D>)
where
    DefaultAllocator: Allocator<D>,
{
    let dif = end - start;
    let l = dif.norm();

    if l < T::default_epsilon() {
        return (u0, start.clone());
    }

    let o = start.clone();
    let r = dif / l;
    let o2pt = pt - &o;
    let do2ptr = o2pt.dot(&r);

    if do2ptr < T::zero() {
        (u0, start.clone())
    } else if do2ptr > l {
        (u1, end.clone())
    } else {
        (u0 + (u1 - u0) * do2ptr / l, (r * do2ptr + o.coords).into())
    }
}
