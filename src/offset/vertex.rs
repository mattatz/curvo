use nalgebra::Point2;

use crate::misc::FloatingPoint;

/// vertex variant
#[derive(Debug, Clone, PartialEq)]
pub enum Vertex<T: FloatingPoint> {
    Point(Point2<T>),
    Intersection(Point2<T>),
}

impl<T: FloatingPoint, P: geo::CoordNum> From<geo::Point<P>> for Vertex<T> {
    fn from(p: geo::Point<P>) -> Self {
        let x = p.x().to_f64().unwrap();
        let y = p.y().to_f64().unwrap();
        Vertex::Point(Point2::new(
            T::from_f64(x).unwrap(),
            T::from_f64(y).unwrap(),
        ))
    }
}

impl<T: FloatingPoint> From<Vertex<T>> for Point2<T> {
    fn from(v: Vertex<T>) -> Self {
        match v {
            Vertex::Point(p) => p,
            Vertex::Intersection(p) => p,
        }
    }
}

impl<T: FloatingPoint> Vertex<T> {
    pub fn inner(&self) -> &Point2<T> {
        match self {
            Vertex::Point(p) => p,
            Vertex::Intersection(p) => p,
        }
    }
}
