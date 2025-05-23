use std::borrow::Cow;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};
use rand::Rng;

use crate::{curve::NurbsCurve, misc::FloatingPoint, split::Split};

use super::{BoundingBox, BoundingBoxTree};

/// A struct representing a bounding box tree with curve in D space.
#[derive(Clone)]
pub struct CurveBoundingBoxTree<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    curve: Cow<'a, NurbsCurve<T, D>>,
    tolerance: T,
}

impl<'a, T: FloatingPoint, D: DimName> CurveBoundingBoxTree<'a, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Create a new bounding box tree from a curve.
    pub fn new(curve: &'a NurbsCurve<T, D>, tolerance: Option<T>) -> Self {
        let tol = tolerance.unwrap_or_else(|| {
            let i = curve.knots_domain_interval();
            i / T::from_usize(64).unwrap()
        });
        Self {
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
}

impl<T: FloatingPoint, D: DimName> BoundingBoxTree<T, D> for CurveBoundingBoxTree<'_, T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Check if the curve is dividable or not.
    fn is_dividable(&self) -> bool {
        let interval = self.curve.knots_domain_interval();
        interval > self.tolerance || {
            match self.curve.degree() {
                1 => {
                    // Avoid degeneracy case for polyline curves as following:
                    //   |     /
                    //   |    /
                    // --+--+------- <- intersected at 2 points
                    //   | /
                    //   |/
                    // In the case of a polyline, it may not be possible to detect two intersections when it crosses near a control point.
                    // Therefore, when working with polylines, it is advisable to impose a constraint such that the bounding box contains two control points.
                    interval >= T::from_f64(1e-4).unwrap() && self.curve.control_points().len() >= 3
                }
                _ => false,
            }
        }
    }

    /// Try to divide the curve into two parts.
    fn try_divide(&self) -> anyhow::Result<(Self, Self)> {
        let (min, max) = self.curve.knots_domain();
        let interval = max - min;
        let mid = (min + max) / T::from_usize(2).unwrap();

        let mut rng = rand::rng();
        let r = interval * T::from_f64(1e-1 * (rng.random::<f64>() - 0.5)).unwrap();
        // let r = T::zero(); // non random

        let (head, tail) = self.curve.try_split(mid + r)?;
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
    fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>> {
        self.curve.as_ref().into()
    }
}
