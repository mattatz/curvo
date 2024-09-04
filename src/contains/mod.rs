pub mod compound_curve;
pub mod curve;

pub use compound_curve::*;
pub use curve::*;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use crate::misc::FloatingPoint;

/// Trait for determining if a point is inside a curve.
pub trait Contains<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    type Option;
    fn contains(&self, point: &OPoint<T, D>, option: Self::Option) -> anyhow::Result<bool>;
}

#[cfg(test)]
mod tests;
