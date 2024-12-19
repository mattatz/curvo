use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::misc::FloatingPoint;

use super::BoundingBoxTree;

pub struct BoundingBoxTraversal<T0, T1, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    T0: BoundingBoxTree<T, D>,
    T1: BoundingBoxTree<T, D>,
{
    pairs: Vec<(T0, T1)>,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<T0, T1, T: FloatingPoint, D: DimName> BoundingBoxTraversal<T0, T1, T, D>
where
    D: DimNameSub<U1>,
    T0: BoundingBoxTree<T, D>,
    T1: BoundingBoxTree<T, D>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Try to traverse bounding box tree pairs to find pairs of intersecting curves.
    pub fn try_traverse(ta: T0, tb: T1) -> anyhow::Result<Self> {
        let mut trees = vec![(ta, tb)];
        let mut pairs = vec![];

        let tol = Some(T::zero());
        // let tol = T::from_f64(-1e-4);

        while let Some((a, b)) = trees.pop() {
            if !a.bounding_box().intersects(&b.bounding_box(), tol) {
                continue;
            }

            let ai = a.is_dividable();
            let bi = b.is_dividable();
            match (ai, bi) {
                (false, false) => {
                    pairs.push((a, b));
                }
                (true, false) => {
                    let (a0, a1) = a.try_divide()?;
                    trees.push((a0, b.clone()));
                    trees.push((a1, b));
                }
                (false, true) => {
                    let (b0, b1) = b.try_divide()?;
                    trees.push((a.clone(), b0));
                    trees.push((a, b1));
                }
                (true, true) => {
                    let (a0, a1) = a.try_divide()?;
                    let (b0, b1) = b.try_divide()?;
                    trees.push((a0.clone(), b0.clone()));
                    trees.push((a1.clone(), b0));
                    trees.push((a0, b1.clone()));
                    trees.push((a1, b1));
                }
            };
        }

        Ok(Self {
            pairs,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn pairs(&self) -> &[(T0, T1)] {
        &self.pairs
    }

    pub fn pairs_iter(&self) -> impl Iterator<Item = &(T0, T1)> {
        self.pairs.iter()
    }

    pub fn into_pairs(self) -> Vec<(T0, T1)> {
        self.pairs
    }

    pub fn into_pairs_iter(self) -> impl Iterator<Item = (T0, T1)> {
        self.pairs.into_iter()
    }
}
