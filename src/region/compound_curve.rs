use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

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

    /// Returns the total length of the compound curve.
    pub fn try_length(&self) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let lengthes: anyhow::Result<Vec<T>> =
            self.spans.iter().map(|span| span.try_length()).collect();
        let total = lengthes?.iter().fold(T::zero(), |a, b| a + *b);
        Ok(total)
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

impl<T: FloatingPoint, D: DimName> From<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from(value: NurbsCurve<T, D>) -> Self {
        Self::new(vec![value])
    }
}
