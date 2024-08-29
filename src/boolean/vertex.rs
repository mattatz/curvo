use nalgebra::Point2;

use crate::misc::FloatingPoint;

#[derive(Debug, Clone)]
pub struct Vertex<T: FloatingPoint> {
    position: Point2<T>,
    parameter: T,
}

impl<T: FloatingPoint> Vertex<T> {
    pub fn new(position: Point2<T>, parameter: T) -> Self {
        Self {
            position,
            parameter,
        }
    }

    pub fn position(&self) -> &Point2<T> {
        &self.position
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }
}

impl<'a, T: FloatingPoint> From<&'a (Point2<T>, T)> for Vertex<T> {
    fn from(v: &'a (Point2<T>, T)) -> Self {
        Self {
            position: v.0,
            parameter: v.1,
        }
    }
}
