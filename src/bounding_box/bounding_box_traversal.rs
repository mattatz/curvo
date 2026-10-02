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
        let mut a_nodes = Nodes::new(ta);
        let mut b_nodes = Nodes::new(tb);
        let mut trees = vec![(0, 0)];
        let mut pairs = vec![];

        let tol = Some(T::zero());
        // let tol = T::from_f64(-1e-4);

        while let Some((a, b)) = trees.pop() {
            if !a_nodes.nodes[a]
                .bounding_box
                .intersects(&b_nodes.nodes[b].bounding_box, tol)
            {
                continue;
            }

            let ai = a_nodes.nodes[a].dividable;
            let bi = b_nodes.nodes[b].dividable;
            match (ai, bi) {
                (false, false) => {
                    pairs.push((a, b));
                }
                (true, false) => {
                    let (a0, a1) = a_nodes.try_divide(a)?;
                    trees.push((a0, b));
                    trees.push((a1, b));
                }
                (false, true) => {
                    let (b0, b1) = b_nodes.try_divide(b)?;
                    trees.push((a, b0));
                    trees.push((a, b1));
                }
                (true, true) => {
                    let (a0, a1) = a_nodes.try_divide(a)?;
                    let (b0, b1) = b_nodes.try_divide(b)?;
                    trees.push((a0, b0));
                    trees.push((a1, b0));
                    trees.push((a0, b1));
                    trees.push((a1, b1));
                }
            };
        }

        Ok(Self {
            pairs: pairs
                .into_iter()
                .map(|(a, b)| (a_nodes.nodes[a].tree.clone(), b_nodes.nodes[b].tree.clone()))
                .collect(),
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

/// A node of a bounding box tree as the traversal meets it, with what the traversal asks of it
/// again for every node of the other tree it is paired with.
struct Node<N, T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    tree: N,
    bounding_box: BoundingBox<T, DimNameDiff<D, U1>>,
    dividable: bool,
    children: Option<(usize, usize)>,
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
    fn new(root: N) -> Self {
        let mut nodes = Self { nodes: vec![] };
        nodes.push(root);
        nodes
    }

    fn push(&mut self, tree: N) -> usize {
        self.nodes.push(Node {
            bounding_box: tree.bounding_box(),
            dividable: tree.is_dividable(),
            children: None,
            tree,
        });
        self.nodes.len() - 1
    }

    /// The two halves of the node at `index`, dividing it if it has not been divided yet.
    fn try_divide(&mut self, index: usize) -> anyhow::Result<(usize, usize)> {
        if let Some(children) = self.nodes[index].children {
            return Ok(children);
        }
        let (head, tail) = self.nodes[index].tree.try_divide()?;
        let children = (self.push(head), self.push(tail));
        self.nodes[index].children = Some(children);
        Ok(children)
    }
}
