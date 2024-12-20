use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use super::FloatingPoint;

/// Transpose control points
pub fn transpose_control_points<T: FloatingPoint, D: DimName>(
    points: &Vec<Vec<OPoint<T, D>>>,
) -> Vec<Vec<OPoint<T, D>>>
where
    DefaultAllocator: Allocator<D>,
{
    let mut transposed = vec![vec![]; points[0].len()];
    points.iter().for_each(|row| {
        row.iter().enumerate().for_each(|(j, p)| {
            transposed[j].push(p.clone());
        })
    });
    transposed
}
