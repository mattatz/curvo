use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{bounding_box::BoundingBox, misc::FloatingPoint};

/// A trait representing a bounding box tree.
pub trait BoundingBoxTree<T: FloatingPoint, D: DimName>: Clone
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    Self: Sized,
{
    fn is_dividable(&self) -> bool;
    fn try_divide(&self) -> anyhow::Result<(Self, Self)>;
    fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>>;
}
