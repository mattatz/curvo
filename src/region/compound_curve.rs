use std::cmp::Ordering;

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
    pub fn new(spans: Vec<NurbsCurve<T, D>>) -> Self
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let eps = T::from_f64(1e-5).unwrap();

        // Ensure the adjacent spans are connected in the forward direction.
        let mut curves = spans.clone();
        let mut connected = vec![curves.remove(0)];
        while !curves.is_empty() {
            let current = connected.len() - 1;
            let last = &connected[current];
            let found = curves
                .iter()
                .enumerate()
                .find_map(|(i, c)| Direction::new(last, c, eps).map(|direction| (i, direction)));
            match found {
                Some((index, direction)) => {
                    let next = curves.remove(index);
                    match direction {
                        Direction::Forward => {
                            connected.push(next);
                        }
                        Direction::Backward => {
                            connected.insert(current, next);
                        }
                        Direction::Facing => {
                            connected.push(next.inverse());
                        }
                        Direction::Opposite => {
                            if current == 0 {
                                connected.insert(current, next.inverse());
                            } else {
                                println!("Cannot handle opposite direction");
                            }
                        }
                    }
                }
                None => {
                    println!("No connection found");
                    break;
                }
            }
        }

        Self { spans: connected }
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
#[derive(Clone, Copy, Debug)]
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
    ) -> Option<Self>
    where
        DefaultAllocator: Allocator<D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let ad = a.knots_domain();
        let bd = b.knots_domain();
        let (a0, a1) = (a.point_at(ad.0), a.point_at(ad.1));
        let (b0, b1) = (b.point_at(bd.0), b.point_at(bd.1));
        let directions = [
            ((&a1 - &b0).norm(), Self::Forward),
            ((&a0 - &b1).norm(), Self::Backward),
            ((&a1 - &b1).norm(), Self::Facing),
            ((&a0 - &b0).norm(), Self::Opposite),
        ];

        directions
            .iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal))
            .and_then(|min| if min.0 < epsilon { Some(min.1) } else { None })
    }
}
