use nalgebra::Point2;
use num_traits::NumCast;

use crate::misc::FloatingPoint;

/// Convert a line segment to a geo::Line
pub fn to_line_helper<T: FloatingPoint>(p0: &Point2<T>, p1: &Point2<T>) -> geo::Line {
    let x0 = <f64 as NumCast>::from(p0.x).unwrap();
    let y0 = <f64 as NumCast>::from(p0.y).unwrap();
    let x1 = <f64 as NumCast>::from(p1.x).unwrap();
    let y1 = <f64 as NumCast>::from(p1.y).unwrap();
    geo::Line::new(geo::coord! { x: x0, y: y0 }, geo::coord! { x: x1, y: y1 })
}
