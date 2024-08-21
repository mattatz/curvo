use nalgebra::{allocator::Allocator, DefaultAllocator, DimName};

use crate::{curve::NurbsCurve, misc::FloatingPoint};

#[derive(Clone, Debug)]
pub struct CompoundCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    spans: Vec<NurbsCurve<T, D>>,
}

impl<T: FloatingPoint, D: DimName> CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(spans: Vec<NurbsCurve<T, D>>) -> Self {
        Self { spans }
    }

    pub fn spans(&self) -> &[NurbsCurve<T, D>] {
        &self.spans
    }

    pub fn spans_mut(&mut self) -> &mut [NurbsCurve<T, D>] {
        &mut self.spans
    }
}

impl<T: FloatingPoint, D: DimName> FromIterator<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = NurbsCurve<T, D>>>(iter: I) -> Self {
        Self {
            spans: iter.into_iter().collect(),
        }
    }
}
