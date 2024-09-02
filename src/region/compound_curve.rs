use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Invertible},
};

/// A struct representing a compound curve.
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
    pub fn new(mut spans: Vec<NurbsCurve<T, D>>) -> Self
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let eps = T::from_f64(1e-5).unwrap();

        // Ensure the adjacent spans are connected in the forward direction.
        let n = spans.len();
        for i in 0..(n - 1) {
            let a = &spans[i];
            let b = &spans[i + 1];
            let direction = Direction::new(a, b, eps);
            match direction {
                Direction::Forward => {}
                Direction::Backward => {
                    spans[i].invert();
                    spans[i + 1].invert();
                }
                Direction::Facing => {
                    spans[i + 1].invert();
                }
                Direction::Opposite => {
                    spans[i].invert();
                }
            }
        }

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
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: NurbsCurve<T, D>) -> Self {
        Self::new(vec![value])
    }
}

/// Direction of the two connected curves.
#[derive(Clone, Debug)]
enum Direction {
    Forward,  // -> ->
    Backward, // <- <-
    Facing,   // -> <-
    Opposite, // <- ->
}

impl Direction {
    fn new<T: FloatingPoint, D: DimName>(
        a: &NurbsCurve<T, D>,
        b: &NurbsCurve<T, D>,
        epsilon: T,
    ) -> Self
    where
        DefaultAllocator: Allocator<D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let ad = a.knots_domain();
        let bd = b.knots_domain();
        let (a0, a1) = (a.point_at(ad.0), a.point_at(ad.1));
        let (b0, b1) = (b.point_at(bd.0), b.point_at(bd.1));
        let d10 = &a1 - &b0;
        if d10.norm() < epsilon {
            return Self::Forward;
        }
        let d01 = &a0 - &b1;
        if d01.norm() < epsilon {
            return Self::Backward;
        }
        let d11 = &a1 - &b1;
        if d11.norm() < epsilon {
            return Self::Facing;
        }
        Self::Opposite
    }
}
