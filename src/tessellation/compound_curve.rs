use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Point2, U1,
};

use crate::{
    misc::FloatingPoint,
    region::{CompoundCurve, Region},
};

use super::Tessellation;

impl<T: FloatingPoint, D: DimName> Tessellation for CompoundCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Option = Option<T>;
    type Output = Vec<OPoint<T, DimNameDiff<D, U1>>>;

    fn tessellate(&self, tolerance: Option<T>) -> Self::Output {
        let tessellated = self
            .spans()
            .iter()
            .flat_map(|span| span.tessellate(tolerance))
            .collect_vec();
        tessellated
    }
}
