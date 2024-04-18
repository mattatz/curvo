use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, RealField};

pub fn three_points_are_flat<T: RealField + Copy, D: DimName>(
    p1: &OPoint<T, D>,
    p2: &OPoint<T, D>,
    p3: &OPoint<T, D>,
    tolerance: T,
) -> bool
where
    DefaultAllocator: Allocator<T, D>,
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
