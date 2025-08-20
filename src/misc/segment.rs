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
        let a = plane.constant() - plane.normal().dot(&self.a.coords);
        let b = plane.normal().dot(&dir);
        let bcoord = a / b;
        let dir_norm = dir.norm();

        if b.relative_eq(&T::zero(), T::default_epsilon(), T::default_max_relative())
            || bcoord * dir_norm <= epsilon
            || bcoord * dir_norm >= dir_norm - epsilon
        {
            if a >= T::zero() {
                (SplitResult::Negative, None)
            } else {
                (SplitResult::Positive, None)
            }
        } else {
            let intersection = self.a + dir * bcoord;
            let s1 = Segment::new(self.a, intersection);
            let s2 = Segment::new(intersection, self.b);
            if a >= T::zero() {
                (SplitResult::Pair(s1, s2), Some((intersection, bcoord)))
            } else {
                (SplitResult::Pair(s2, s1), Some((intersection, bcoord)))
            }
        }
    }
}
