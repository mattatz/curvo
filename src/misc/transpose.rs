use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use super::FloatingPoint;

/// Transpose control points
pub fn transpose_control_points<T: FloatingPoint, D: DimName>(
    points: &[Vec<OPoint<T, D>>],
) -> Vec<Vec<OPoint<T, D>>>
where
    DefaultAllocator: Allocator<D>,
{
    let mut transposed: Vec<Vec<_>> = (0..points[0].len())
        .map(|_| Vec::with_capacity(points.len()))
        .collect();
    points.iter().for_each(|row| {
        row.iter().enumerate().for_each(|(j, p)| {
            transposed[j].push(p.clone());
        })
    });
    transposed
}
