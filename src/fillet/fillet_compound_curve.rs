use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub, OPoint,
    OVector, Rotation3, Unit, Vector3, U1,
};

use crate::{
    curve::NurbsCurve,
    fillet::{segment::Segment, Fillet, FilletRadiusOption, FilletRadiusParameterOption},
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

        todo!()
    }
}
