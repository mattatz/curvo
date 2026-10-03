use std::borrow::Cow;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, GoldenRatioSequence},
    split::Split,
};

use super::{BoundingBox, BoundingBoxTree};

/// A struct representing a bounding box tree with curve in D space.
#[derive(Clone)]
pub struct CurveBoundingBoxTree<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    curve: Cow<'a, NurbsCurve<T, D>>,
    tolerance: T,
    /// Where this node and those under it are divided, off the middle of their domain.
    sequence: GoldenRatioSequence,
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
            sequence: GoldenRatioSequence::default(),
        }
    }

    /// Create a new bounding box tree from a curve, divided down to one `divisions`-th of its
    /// domain.
    pub fn with_divisions(curve: &'a NurbsCurve<T, D>, divisions: usize) -> Self {
        let tolerance = curve.knots_domain_interval() / T::from_usize(divisions).unwrap();
        Self::new(curve, Some(tolerance))
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

        // Off the exact middle, where regular geometry tends to have its intersections, by up
        // to a twentieth of the domain either way. The offset is not random: the tree of a curve
        // is the same every time, and so is what is found with it.
        let mut sequence = self.sequence;
        let r = interval * T::from_f64(1e-1 * (sequence.next() - 0.5)).unwrap();

        // the two halves go on from different places in the sequence
        let head_sequence = sequence;
        sequence.next();

        let (head, tail) = self.curve.try_split(mid + r)?;
        Ok((
            Self {
                curve: Cow::Owned(head),
                tolerance: self.tolerance,
                sequence: head_sequence,
            },
            Self {
                curve: Cow::Owned(tail),
                tolerance: self.tolerance,
                sequence,
            },
        ))
    }

    /// Get the bounding box of the curve.
    fn bounding_box(&self) -> BoundingBox<T, DimNameDiff<D, U1>> {
        self.curve.as_ref().into()
    }
}
