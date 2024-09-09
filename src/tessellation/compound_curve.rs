use std::f64::consts::E;

use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{misc::FloatingPoint, region::CompoundCurve};

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
        let closed = self.is_closed();
        if closed {
            let pts = self
                .spans()
                .iter()
                .flat_map(|span| {
                    let mut tess = span.tessellate(tolerance);
                    tess.pop();
                    tess
                })
                .collect_vec();
            let n = pts.len();
            pts.into_iter().cycle().take(n + 1).collect_vec()
        } else {
            let m = self.spans().len();
            self.spans()
                .iter()
                .enumerate()
                .flat_map(|(i, span)| {
                    let mut tess = span.tessellate(tolerance);
                    if i != m - 1 {
                        tess.pop();
                    }
                    tess
                })
                .collect_vec()
        }
    }
}
