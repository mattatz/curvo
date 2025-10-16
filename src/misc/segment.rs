use nalgebra::Point3;

use crate::{misc::Plane, prelude::FloatingPoint};

/// A segment in 3D space.
pub struct Segment<T: FloatingPoint> {
    pub a: Point3<T>,
    pub b: Point3<T>,
}

#[derive(Debug, Clone, Copy)]
pub enum SplitResult<T> {
    /// The split operation yield two results: one lying on the negative half-space of the plane
    /// and the second lying on the positive half-space of the plane.
    Pair(T, T),
    /// The shape being split is fully contained in the negative half-space of the plane.
    Negative,
    /// The shape being split is fully contained in the positive half-space of the plane.
    Positive,
    /// The segment is parallel to the plane.
    Parallel,
}

impl<T: FloatingPoint> Segment<T> {
    pub fn new(a: Point3<T>, b: Point3<T>) -> Self {
        Self { a, b }
    }

    /// Split a segment with a plane.
    pub fn split(
        &self,
        plane: &Plane<T>,
        epsilon: T,
    ) -> (SplitResult<Self>, Option<(Point3<T>, T)>) {
        let dir = self.b - self.a;
        let dir_norm = dir.norm();

        let d_a = plane.signed_distance(&self.a);
        let d_b = plane.signed_distance(&self.b);
        if d_a > T::zero() && d_b > T::zero() {
            return (SplitResult::Positive, None);
        } else if d_a < T::zero() && d_b < T::zero() {
            return (SplitResult::Negative, None);
        }

        let denom = d_a - d_b;
        if denom.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative()) {
            return (SplitResult::Parallel, None);
        }

        let t = d_a / denom;
        if (dir_norm * t).abs() <= epsilon {
            return (SplitResult::Parallel, None);
        }

        let intersection = self.a + dir * t;
        let s1 = Segment::new(self.a, intersection);
        let s2 = Segment::new(intersection, self.b);
        if d_a >= T::zero() {
            (SplitResult::Pair(s1, s2), Some((intersection, t)))
        } else {
            (SplitResult::Pair(s2, s1), Some((intersection, t)))
        }
    }
}
