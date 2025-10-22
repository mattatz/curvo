use geo::LineIntersection;
use nalgebra::{Point2, Vector2};
use num_traits::NumCast;

use crate::{curve::NurbsCurve2D, misc::FloatingPoint};

/// Convert a line segment to a geo::Line
pub fn to_line_helper<T: FloatingPoint>(p0: &Point2<T>, p1: &Point2<T>) -> geo::Line {
    let x0 = <f64 as NumCast>::from(p0.x).unwrap();
    let y0 = <f64 as NumCast>::from(p0.y).unwrap();
    let x1 = <f64 as NumCast>::from(p1.x).unwrap();
    let y1 = <f64 as NumCast>::from(p1.y).unwrap();
    geo::Line::new(geo::coord! { x: x0, y: y0 }, geo::coord! { x: x1, y: y1 })
}

/// find the intersection of the line v0-v1 & v2-v3
pub fn sharp_corner_intersection<T: FloatingPoint>(
    vertices: [&Point2<T>; 4],
    delta: T,
) -> Option<Point2<T>> {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];

    let d0 = (v1 - v0).normalize() * delta;
    let l0 = to_line_helper(v0, &(v1 + d0));
    let d1 = (v3 - v2).normalize() * delta;
    let l1 = to_line_helper(&(v2 - d1), v3);

    let it = geo::algorithm::line_intersection::line_intersection(l0, l1);
    let it = it
        .and_then(|it| match it {
            LineIntersection::SinglePoint {
                intersection: p,
                is_proper: _,
            } => Some(p),
            _ => None,
        })?;

    Some(Point2::new(
        T::from_f64(it.x).unwrap(),
        T::from_f64(it.y).unwrap(),
    ))
}

/// Create a round corner between two points (v1 & v2)
/// v0---v1
///       \
///        \
///         v2
pub fn round_corner<T: FloatingPoint>(
    v0: &Point2<T>,
    v1: &Point2<T>,
    v2: &Point2<T>,
    distance: T,
) -> anyhow::Result<NurbsCurve2D<T>> {
    let t = (v1 - v0).normalize();
    let sign = distance.signum();
    let n = Vector2::new(t.y, -t.x);
    let d = n * distance;
    let center = v1 - d;

    let d0 = v1 - center;
    let d1 = v2 - center;
    let angle = d0.angle(&d1);
    let angle = angle.abs();
    NurbsCurve2D::try_arc(&center, &(n * sign), &t, distance.abs(), T::zero(), angle)
}

/// Create a smooth corner between two points (v1 & v2)
/// v0---v1
///       \
///        \
///         v2
///         |
///         v3
pub fn smooth_corner<T: FloatingPoint>(
    v0: &Point2<T>,
    v1: &Point2<T>,
    v2: &Point2<T>,
    v3: &Point2<T>,
    distance: T,
) -> anyhow::Result<NurbsCurve2D<T>> {
    let frac_2_3 = T::from_f64(2.0 / 3.0).unwrap();
    let d = distance.abs() * frac_2_3;
    let d10 = if v1 == v0 {
        Vector2::zeros()
    } else {
        (v1 - v0).normalize() * d
    };
    let d32 = if v3 == v2 {
        Vector2::zeros()
    } else {
        (v3 - v2).normalize() * d
    };

    // create arc between v1 and v2
    let bezier = NurbsCurve2D::bezier(&[*v1, v1 + d10, v2 - d32, *v2]);
    Ok(bezier)
}
