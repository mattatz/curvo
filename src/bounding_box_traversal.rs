use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{nurbs_curve::NurbsCurve, prelude::BoundingBoxTree, FloatingPoint};

pub struct BoundingBoxTraversal<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    pairs: Vec<(BoundingBoxTree<'a, T, D>, BoundingBoxTree<'a, T, D>)>,
}

impl<'a, T: FloatingPoint, D: DimName> BoundingBoxTraversal<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    /// Try to traverse bounding box tree pairs to find pairs of intersecting curves.
    pub fn try_traverse(
        a: &'a NurbsCurve<T, D>,
        b: &'a NurbsCurve<T, D>,
        tolerance: Option<T>,
    ) -> anyhow::Result<Self> {
        let ta = BoundingBoxTree::new(a, tolerance);
        let tb = BoundingBoxTree::new(b, tolerance);

        let mut trees = vec![(ta, tb)];
        let mut pairs = vec![];

        while trees.len() > 0 {
            let (a, b) = trees.pop().unwrap();

            if !a.bounding_box().intersects(&b.bounding_box()) {
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

        Ok(Self { pairs })
    }

    pub fn pairs(&self) -> &[(BoundingBoxTree<'a, T, D>, BoundingBoxTree<'a, T, D>)] {
        &self.pairs
    }

    pub fn pairs_iter(
        &self,
    ) -> impl Iterator<Item = &(BoundingBoxTree<'a, T, D>, BoundingBoxTree<'a, T, D>)> {
        self.pairs.iter()
    }
}
