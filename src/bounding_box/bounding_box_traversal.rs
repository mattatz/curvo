use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::misc::FloatingPoint;

use super::{BoundingBox, BoundingBoxTree};

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
        Self::try_traverse_with_tolerance(ta, tb, T::zero())
    }

    /// Try to traverse bounding box tree pairs, where bounding boxes less than `tolerance` apart
    /// along every axis count as overlapping.
    ///
    /// Geometry that comes within a distance of each other without touching, two curves in
    /// planes a rounding error apart, has bounding boxes that do not overlap: the pairs that are
    /// that close are only found with that distance as the tolerance.
    pub fn try_traverse_with_tolerance(ta: T0, tb: T1, tolerance: T) -> anyhow::Result<Self> {
        let mut a_nodes = Nodes::new(ta);
        let mut b_nodes = Nodes::new(tb);
        let mut trees = vec![(0, 0)];
        let mut pairs = vec![];

        // each of the two boxes grows by half the tolerance
        let tol = Some(tolerance / T::from_f64(2.).unwrap());

        while let Some((a, b)) = trees.pop() {
            if !a_nodes
                .bounding_box(a)
                .intersects(b_nodes.bounding_box(b), tol)
            {
                continue;
            }

            match (a_nodes.try_visit(a)?, b_nodes.try_visit(b)?) {
                (Visit::Leaf(a), Visit::Leaf(b)) => {
                    pairs.push((a.clone(), b.clone()));
                }
                (Visit::Children(a0, a1), Visit::Leaf(_)) => {
                    trees.push((a0, b));
                    trees.push((a1, b));
                }
                (Visit::Leaf(_), Visit::Children(b0, b1)) => {
                    trees.push((a, b0));
                    trees.push((a, b1));
                }
                (Visit::Children(a0, a1), Visit::Children(b0, b1)) => {
                    trees.push((a0, b0));
                    trees.push((a1, b0));
                    trees.push((a0, b1));
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

/// A node of a bounding box tree as the traversal meets it. The traversal meets a node once for
/// every node of the other tree it is paired with, so the node keeps its bounding box and, once
/// divided, its halves, instead of computing them again each time.
struct Node<N, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    bounding_box: BoundingBox<T, DimNameDiff<D, U1>>,
    state: State<N>,
}

enum State<N> {
    /// Not divided: a leaf, or a node the traversal has not had to divide yet.
    Whole(N),
    /// Divided into the nodes at these indices. The tree itself is dropped, only its halves are
    /// needed from here on.
    Divided(usize, usize),
}

/// What the traversal finds at a node.
enum Visit<'a, N> {
    /// A node that cannot be divided.
    Leaf(&'a N),
    /// The indices of the halves of a node that can.
    Children(usize, usize),
}

/// The nodes of one bounding box tree, divided on demand and at most once each.
struct Nodes<N, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    nodes: Vec<Node<N, T, D>>,
}

impl<N, T: FloatingPoint, D: DimName> Nodes<N, T, D>
where
    D: DimNameSub<U1>,
    N: BoundingBoxTree<T, D>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// The nodes of the tree under `root`, which is the node at index 0.
    fn new(root: N) -> Self {
        let mut nodes = Self { nodes: vec![] };
        nodes.push(root);
        nodes
    }

    fn push(&mut self, tree: N) -> usize {
        self.nodes.push(Node {
            bounding_box: tree.bounding_box(),
            state: State::Whole(tree),
        });
        self.nodes.len() - 1
    }

    fn bounding_box(&self, index: usize) -> &BoundingBox<T, DimNameDiff<D, U1>> {
        &self.nodes[index].bounding_box
    }

    /// Visit the node at `index`, dividing it if it can be divided and has not been yet.
    fn try_visit(&mut self, index: usize) -> anyhow::Result<Visit<'_, N>> {
        if let State::Whole(tree) = &self.nodes[index].state {
            if tree.is_dividable() {
                let (head, tail) = tree.try_divide()?;
                let (head, tail) = (self.push(head), self.push(tail));
                self.nodes[index].state = State::Divided(head, tail);
            }
        }
        Ok(match &self.nodes[index].state {
            State::Whole(tree) => Visit::Leaf(tree),
            State::Divided(head, tail) => Visit::Children(*head, *tail),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use nalgebra::{Vector1, U2};

    use super::*;

    /// An interval that divides in halves down to a length of 1, and records every division.
    #[derive(Clone)]
    struct Interval {
        min: f64,
        max: f64,
        divided: Rc<RefCell<Vec<(f64, f64)>>>,
    }

    impl Interval {
        fn new(min: f64, max: f64) -> Self {
            Self {
                min,
                max,
                divided: Default::default(),
            }
        }

        fn leaves(&self) -> Vec<(f64, f64)> {
            if !self.is_dividable() {
                return vec![(self.min, self.max)];
            }
            let mid = (self.min + self.max) / 2.;
            let mut leaves = Self::new(self.min, mid).leaves();
            leaves.extend(Self::new(mid, self.max).leaves());
            leaves
        }
    }

    impl BoundingBoxTree<f64, U2> for Interval {
        fn is_dividable(&self) -> bool {
            self.max - self.min > 1.
        }

        fn try_divide(&self) -> anyhow::Result<(Self, Self)> {
            self.divided.borrow_mut().push((self.min, self.max));
            let mid = (self.min + self.max) / 2.;
            let half = |min, max| Self {
                min,
                max,
                divided: self.divided.clone(),
            };
            Ok((half(self.min, mid), half(mid, self.max)))
        }

        fn bounding_box(&self) -> BoundingBox<f64, U1> {
            BoundingBox::new(Vector1::new(self.min), Vector1::new(self.max))
        }
    }

    #[test]
    fn traversal_pairs_the_leaves_that_overlap_and_divides_each_node_once() {
        let a = Interval::new(0., 8.);
        let b = Interval::new(2.5, 6.5);
        let (a_divided, b_divided) = (a.divided.clone(), b.divided.clone());

        let mut expected = vec![];
        for la in a.leaves() {
            for lb in b.leaves() {
                // touching intervals count as overlapping
                if la.0 <= lb.1 && lb.0 <= la.1 {
                    expected.push((la, lb));
                }
            }
        }
        assert!(!expected.is_empty());

        let traversal = BoundingBoxTraversal::try_traverse(a, b).unwrap();
        let mut pairs: Vec<_> = traversal
            .pairs_iter()
            .map(|(a, b)| ((a.min, a.max), (b.min, b.max)))
            .collect();
        let order =
            |x: &((f64, f64), (f64, f64)), y: &((f64, f64), (f64, f64))| x.partial_cmp(y).unwrap();
        pairs.sort_by(order);
        expected.sort_by(order);
        assert_eq!(pairs, expected);

        for divided in [a_divided, b_divided] {
            let mut divided = divided.borrow().clone();
            let count = divided.len();
            divided.sort_by(|x, y| x.partial_cmp(y).unwrap());
            divided.dedup();
            assert_eq!(divided.len(), count, "a node was divided more than once");
        }
    }
}
