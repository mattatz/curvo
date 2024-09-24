use nalgebra::Point2;
use num_traits::NumCast;
use robust::{orient2d, Coord};

use super::FloatingPoint;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum Orientation {
    CounterClockwise,
    Clockwise,
    Collinear,
}

/// Robust orientation test for three points.
/// implementation from geo crate. (https://github.com/georust/geo)
pub fn orientation<T: FloatingPoint>(p: &Point2<T>, q: &Point2<T>, r: &Point2<T>) -> Orientation {
    let orientation = orient2d(
        Coord {
            x: <f64 as NumCast>::from(p.x).unwrap(),
            y: <f64 as NumCast>::from(p.y).unwrap(),
        },
        Coord {
            x: <f64 as NumCast>::from(q.x).unwrap(),
            y: <f64 as NumCast>::from(q.y).unwrap(),
        },
        Coord {
            x: <f64 as NumCast>::from(r.x).unwrap(),
            y: <f64 as NumCast>::from(r.y).unwrap(),
        },
    );

    if orientation < 0. {
        Orientation::Clockwise
    } else if orientation > 0. {
        Orientation::CounterClockwise
    } else {
        Orientation::Collinear
    }
}
