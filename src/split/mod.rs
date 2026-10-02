pub mod split_compound_curve;
pub mod split_nurbs_curve;
pub mod split_nurbs_surface;

pub use split_nurbs_surface::*;

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use crate::{curve::nurbs_curve::refine_knot, knot::KnotVector, misc::FloatingPoint};

/// Split the object into two objects with the given option
pub trait Split
where
    Self: Sized,
{
    type Option;
    fn try_split(&self, option: Self::Option) -> anyhow::Result<(Self, Self)>;
}

/// How to split a curve of `degree` at `u`: its knots are refined until `u` is a knot `degree + 1`
/// times, which leaves the control points and knots of the head followed by those of the tail.
struct SplitAt<T> {
    degree: usize,
    u: T,
    /// How many times `u` is inserted.
    insert: usize,
    /// The number of control points of the head.
    head: usize,
    /// The index, in the refined curve, of the first control point and knot of the tail.
    tail: usize,
}

impl<T: FloatingPoint> SplitAt<T> {
    fn new(knots: &KnotVector<T>, degree: usize, u: T) -> Self {
        let n = knots.len() - degree - 2;
        let s = knots.find_knot_span_index(n, degree, u);
        // The knots already at `u` count towards the `degree + 1`: inserting that many on top of
        // them would leave the head with spans of zero length at its end, where it evaluates to
        // NaN. At the start of the domain the head is empty however it is split, and there the
        // knots are inserted regardless, so that both halves still have control points.
        let (start, _) = knots.domain(degree);
        let multiplicity = if u > start {
            (0..=s).rev().take_while(|i| knots[*i] == u).count()
        } else {
            0
        };
        let insert = (degree + 1).saturating_sub(multiplicity);
        let head = s + 1 - multiplicity;
        Self {
            degree,
            u,
            insert,
            head,
            // past the knots at `u` beyond `degree + 1`, if a malformed curve has any
            tail: head + (multiplicity + insert) - (degree + 1),
        }
    }

    /// The control points and knots of the curve refined for this split.
    fn refine<D: DimName>(
        &self,
        control_points: &[OPoint<T, D>],
        knots: &KnotVector<T>,
    ) -> (Vec<OPoint<T, D>>, Vec<T>)
    where
        DefaultAllocator: Allocator<D>,
    {
        // on the stack for the usual degrees
        const INLINE: usize = 16;
        match self.insert {
            0 => (control_points.to_vec(), knots.to_vec()),
            1..=INLINE => refine_knot(
                self.degree,
                control_points,
                knots,
                &[self.u; INLINE][..self.insert],
            ),
            _ => refine_knot(
                self.degree,
                control_points,
                knots,
                &vec![self.u; self.insert],
            ),
        }
    }

    /// Divide the refined control points into those of the head and those of the tail.
    fn divide_control_points<P>(&self, mut refined: Vec<P>) -> (Vec<P>, Vec<P>) {
        let tail = refined.split_off(self.tail);
        refined.truncate(self.head);
        (refined, tail)
    }

    /// Divide the refined knots into those of the head and those of the tail.
    fn divide_knots(&self, mut refined: Vec<T>) -> (Vec<T>, Vec<T>) {
        let tail = refined[self.tail..].to_vec();
        refined.truncate(self.head + self.degree + 1);
        (refined, tail)
    }
}
