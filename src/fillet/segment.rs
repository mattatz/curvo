use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, OVector};

use crate::misc::FloatingPoint;

/// Segment of a curve
#[derive(Debug, Clone)]
pub struct Segment<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    start: OPoint<T, D>,
    end: OPoint<T, D>,
    tangent: OVector<T, D>,
    length: T,
}

impl<T: FloatingPoint, D: DimName> Segment<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(start: OPoint<T, D>, end: OPoint<T, D>) -> Self {
        let tangent = &end - &start;
        let length = tangent.norm();
        Self {
            start,
            end,
            tangent: tangent.normalize(),
            length,
        }
    }

    pub fn start(&self) -> &OPoint<T, D> {
        &self.start
    }

    pub fn end(&self) -> &OPoint<T, D> {
        &self.end
    }

    pub fn tangent(&self) -> &OVector<T, D> {
        &self.tangent
    }

    pub fn length(&self) -> T {
        self.length
    }

    /// Trim the segment by a given length
    pub fn trim(&self, length: T) -> (Self, Self) {
        let t = self.tangent() * length;
        let end = self.start() + &t;
        (
            Self::new(self.start().clone(), end.clone()),
            Self::new(end, self.end().clone()),
        )
    }
}
