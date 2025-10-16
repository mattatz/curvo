use std::borrow::Cow;

use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1,
};

use crate::{
    misc::{FloatingPoint, Plane},
    split::{Split, SplitSurfaceOption},
    surface::{NurbsSurface, UVDirection},
};

use super::{BoundingBox, BoundingBoxTree};

/// A struct representing a bounding box tree with surface in D space.
#[derive(Clone)]
pub struct SurfaceBoundingBoxTree<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    surface: Cow<'a, NurbsSurface<T, D>>,
    tolerance: (T, T),
    direction: UVDirection,
}

impl<'a, T: FloatingPoint, D: DimName> SurfaceBoundingBoxTree<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Create a new bounding box tree from a surface.
    pub fn new(
        surface: &'a NurbsSurface<T, D>,
        direction: UVDirection,
        tolerance: Option<(T, T)>,
    ) -> Self {
        let tol = tolerance.unwrap_or_else(|| {
            let (u, v) = surface.knots_domain_interval();
            let div = T::from_usize(64).unwrap();
            (u / div, v / div)
        });
        Self {
            surface: Cow::Borrowed(surface),
            tolerance: tol,
            direction,
        }
    }

    pub fn surface(&self) -> &NurbsSurface<T, D> {
        self.surface.as_ref()
    }

    pub fn surface_owned(self) -> NurbsSurface<T, D> {
        self.surface.into_owned()
    }
}

impl<T: FloatingPoint, D: DimName> BoundingBoxTree<T, D> for SurfaceBoundingBoxTree<'_, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Check if the surface is dividable or not.
    fn is_dividable(&self) -> bool {
        let interval = self.surface.knots_domain_interval();
        let (utol, vtol) = self.tolerance;
        interval.0 > utol || interval.1 > vtol
    }

    /// Try to divide the surface into two parts.
    fn try_divide(&self) -> anyhow::Result<(Self, Self)> {
        let div = T::from_f64(0.5).unwrap();
        let t = self.surface().knots_domain_at(self.direction);
        let mid = (t.0 + t.1) * div;
        let (l, r) = self
            .surface()
            .try_split(SplitSurfaceOption::new(mid, self.direction))?;
        let next = self.direction.opposite();
        Ok((
            Self {
                surface: Cow::Owned(l),
                tolerance: self.tolerance,
                direction: next,
            },
            Self {
                surface: Cow::Owned(r),
                tolerance: self.tolerance,
                direction: next,
            },
        ))
    }

    /// Get the bounding box of the surface.
    fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>> {
        self.surface.as_ref().into()
    }
}

impl<'a, T: FloatingPoint> SurfaceBoundingBoxTree<'a, T, Const<4>> {
    /// Recursively traverse all leaf nodes from a bounding box tree that intersect the plane
    pub fn traverse_leaf_nodes_with_plane(&self, plane: &Plane<T>) -> Vec<Self> {
        let bbox = self.bounding_box();
        let corners = bbox.corners();

        // Check if bbox straddles the plane
        let distances = corners
            .iter()
            .map(|p| plane.signed_distance(p))
            .collect_vec();
        let has_positive = distances.iter().any(|&d| d > T::zero());
        let has_negative = distances.iter().any(|&d| d < T::zero());

        if !has_positive || !has_negative {
            // Bbox is entirely on one side of the plane
            return vec![];
        }

        if self.is_dividable() {
            if let Ok((left, right)) = self.try_divide() {
                let mut nodes = left.traverse_leaf_nodes_with_plane(plane);
                nodes.extend(right.traverse_leaf_nodes_with_plane(plane));
                nodes
            } else {
                vec![self.clone()]
            }
        } else {
            vec![self.clone()]
        }
    }
}
