use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, U1,
};

use crate::{
    fillet::{Fillet, FilletRadiusOption},
    misc::FloatingPoint,
    region::CompoundCurve,
};

impl<T: FloatingPoint, D: DimName> Fillet<FilletRadiusOption<T>> for CompoundCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Output = anyhow::Result<CompoundCurve<T, D>>;

    /// Fillet the sharp corners of the curve with a given radius
    fn fillet(&self, option: FilletRadiusOption<T>) -> Self::Output {
        let radius = option.radius();

        let spans = self
            .spans()
            .iter()
            .map(|span| span.fillet(FilletRadiusOption::new(radius)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let is_closed = self.is_closed(None);

        /*
        // flatten the spans
        let spans = spans.into_iter().flat_map(|c| c.into_spans()).collect_vec();
        if is_closed {
            let last = spans
                .last()
                .ok_or(anyhow::anyhow!("Closed curve must have at least one span"))?;
            let head = spans
                .first()
                .ok_or(anyhow::anyhow!("Closed curve must have at least one span"))?;
            if last.degree() == 1 && last.degree() == head.degree() {
                // create fillet arc
            }
        }
        Ok(CompoundCurve::new_unchecked_aligned(spans))
        */

        todo!()
    }
}
