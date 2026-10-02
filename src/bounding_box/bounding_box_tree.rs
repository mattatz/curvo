use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1,
};

use crate::{
    bounding_box::BoundingBox,
    misc::{FloatingPoint, Plane},
};

/// A trait representing a bounding box tree.
pub trait BoundingBoxTree<T: FloatingPoint, D: DimName>: Clone
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    Self: Sized,
{
    fn is_dividable(&self) -> bool;
    fn try_divide(&self) -> anyhow::Result<(Self, Self)>;
    fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>>;
}

/// Collect the leaves of a bounding box tree in 3D that reach within `tolerance` of a plane: those
/// whose bounding box is not all further than that on one side of it. With a tolerance, a curve
/// or a surface that ends on the plane or touches it is not left out.
pub fn leaves_reaching_plane<T, N>(tree: N, plane: &Plane<T>, tolerance: T) -> Vec<N>
where
    T: FloatingPoint,
    N: BoundingBoxTree<T, Const<4>>,
{
    let corners = tree.bounding_box().corners();
    let distances = || corners.iter().map(|corner| plane.signed_distance(corner));
    let above = distances().any(|d| d >= -tolerance);
    let below = distances().any(|d| d <= tolerance);
    if !above || !below {
        // all on one side of the plane
        return vec![];
    }

    if tree.is_dividable() {
        if let Ok((left, right)) = tree.try_divide() {
            let mut leaves = leaves_reaching_plane(left, plane, tolerance);
            leaves.extend(leaves_reaching_plane(right, plane, tolerance));
            return leaves;
        }
    }
    vec![tree]
}
