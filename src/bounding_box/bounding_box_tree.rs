use std::borrow::Cow;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{bounding_box::BoundingBox, curve::nurbs_curve::NurbsCurve, misc::FloatingPoint};

/// A struct representing a bounding box tree in D space.
#[derive(Clone)]
pub struct BoundingBoxTree<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    curve: Cow<'a, NurbsCurve<T, D>>,
    tolerance: T,
}

impl<'a, T: FloatingPoint, D: DimName> BoundingBoxTree<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    /// Create a new bounding box tree from a curve.
    pub fn new(curve: &'a NurbsCurve<T, D>, tolerance: Option<T>) -> Self {
        let tol = tolerance.unwrap_or_else(|| {
            let i = curve.knots_domain_interval();
            i / T::from_usize(64).unwrap()
        });
        BoundingBoxTree {
            curve: Cow::Borrowed(curve),
            tolerance: tol,
        }
    }

    pub fn curve(&self) -> &NurbsCurve<T, D> {
        self.curve.as_ref()
    }

    pub fn curve_owned(self) -> NurbsCurve<T, D> {
        self.curve.into_owned()
    }

    /// Check if the curve is dividable or not.
    pub fn is_dividable(&self) -> bool {
        let i = self.curve.knots_domain_interval();
        i > self.tolerance
    }

    /// Try to divide the curve into two parts.
    pub fn try_divide(&self) -> anyhow::Result<(Self, Self)> {
        let (min, max) = self.curve.knots_domain();
        let interval = max - min;
        let (head, tail) = self
            .curve
            .try_trim(min + interval / T::from_usize(2).unwrap())?;
        Ok((
            Self {
                curve: Cow::Owned(head),
                tolerance: self.tolerance,
            },
            Self {
                curve: Cow::Owned(tail),
                tolerance: self.tolerance,
            },
        ))
    }

    /// Get the bounding box of the curve.
    pub fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>> {
        self.curve.as_ref().into()
    }
}
