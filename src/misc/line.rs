use nalgebra::{Const, Point2, Vector2};

use crate::prelude::BoundingBox;

use super::{orientation, FloatingPoint, Orientation};

/// A struct representing a line in 2D space.
pub struct Line<T: FloatingPoint> {
    start: Point2<T>,
    end: Point2<T>,
}

impl<T: FloatingPoint> Line<T> {
    pub fn new(start: Point2<T>, end: Point2<T>) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> &Point2<T> {
        &self.start
    }

    pub fn end(&self) -> &Point2<T> {
        &self.end
    }

    pub fn tangent(&self) -> Vector2<T> {
        self.end - self.start
    }

    /// Robust intersection test between two lines.
    pub fn intersects(&self, other: &Line<T>) -> bool {
        if !self
            .bounding_rect()
            .intersects(&other.bounding_rect(), None)
        {
            return false;
        }

        let p_q1 = orientation(self.start(), self.end(), other.start());
        let p_q2 = orientation(self.start(), self.end(), other.end());

        if matches!(
            (p_q1, p_q2),
            (Orientation::Clockwise, Orientation::Clockwise)
                | (Orientation::CounterClockwise, Orientation::CounterClockwise)
        ) {
            return false;
        }

        let q_p1 = orientation(other.start(), other.end(), self.start());
        let q_p2 = orientation(other.start(), other.end(), self.end());

        // println!("{:?} {:?} {:?} {:?}", p_q1, p_q2, q_p1, q_p2);

        if matches!(
            (q_p1, q_p2),
            (Orientation::Clockwise, Orientation::Clockwise)
                | (Orientation::CounterClockwise, Orientation::CounterClockwise)
        ) {
            return false;
        }

        if matches!(
            (p_q1, p_q2, q_p1, q_p2),
            (
                Orientation::Collinear,
                Orientation::Collinear,
                Orientation::Collinear,
                Orientation::Collinear
            )
        ) {
            return false;
        }

        true
    }

    /// Returns the bounding box of the line.
    pub fn bounding_rect(&self) -> BoundingBox<T, Const<2>> {
        BoundingBox::new(self.start.coords, self.end.coords)
    }
}
