pub mod compound_curve;
pub mod curve;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use crate::misc::FloatingPoint;

/// Trait for determining if a point is inside a curve.
pub trait Contains<T: FloatingPoint, D: DimName, O>
where
    DefaultAllocator: Allocator<D>,
{
    fn contains(&self, point: &OPoint<T, D>, option: O) -> anyhow::Result<bool>;
}

#[cfg(test)]
mod tests;
