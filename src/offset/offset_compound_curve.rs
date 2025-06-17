use geo::LineIntersection;
use itertools::Itertools;
use nalgebra::Point2;

use crate::{
    curve::NurbsCurve2D,
    misc::FloatingPoint,
    offset::{
        helper::{round_corner, sharp_corner_intersection, smooth_corner, to_line_helper},
        CurveOffsetCornerType, CurveOffsetOption, Offset,
    },
    region::CompoundCurve2D,
};

impl<'a, T> Offset<'a, T> for CompoundCurve2D<T>
where
    T: FloatingPoint,
{
    type Output = anyhow::Result<Vec<CompoundCurve2D<T>>>;
    type Option = CurveOffsetOption<T>;

    fn offset(&'a self, option: Self::Option) -> Self::Output {
        let offset = self
            .spans()
            .iter()
            .map(|span| span.offset(option.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;

        if matches!(option.corner_type(), CurveOffsetCornerType::None) {
            return Ok(offset.into_iter().flatten().collect());
        }

        let is_closed = self.is_closed(None);

        let corners = offset
            .windows(2)
            .map(|window| {
                let w0 = &window[0];
                let w1 = &window[1];
                find_corner(w0, w1, &option)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let last_corner = if is_closed {
            let last = offset.last();
            let head = offset.first();
            match (last, head) {
                (Some(last), Some(head)) => find_corner(last, head, &option)?,
                _ => None,
            }
        } else {
            None
        };

        let corners = corners.into_iter().chain(vec![last_corner]).collect_vec();

        let n = corners.len();
        let curves = offset.into_iter().enumerate().map(|(i, o)| {
            let prev_corner = &corners[(i + n - 1) % n];
            let next_corner = &corners[i % n];
            match (prev_corner, next_corner) {
                (Some(Corner::Intersection(prev)), Some(next)) => {
                    match next {
                        Corner::Intersection(next) => {
                            // create a trimmed curve between prev & next intersections
                            let trimmed = trim_polyline(
                                o[0].spans().first().unwrap(),
                                Some(*prev),
                                Some(*next),
                            );
                            vec![trimmed]
                        }
                        Corner::Curve(corner) => {
                            // create a trimmed curve from prev intersection
                            let trimmed =
                                trim_polyline(o[0].spans().first().unwrap(), Some(*prev), None);
                            vec![trimmed, corner.clone()]
                        }
                    }
                }
                (_, Some(next)) => match next {
                    Corner::Intersection(next) => {
                        // create a trimmed curve from prev intersection
                        let trimmed =
                            trim_polyline(o[0].spans().first().unwrap(), None, Some(*next));
                        vec![trimmed]
                    }
                    Corner::Curve(corner) => o
                        .into_iter()
                        .flat_map(|c| c.into_spans())
                        .chain(vec![corner.clone()])
                        .collect_vec(),
                },
                (_, None) => o.into_iter().flat_map(|c| c.into_spans()).collect_vec(),
            }
        });

        Ok(vec![CompoundCurve2D::new_unchecked_aligned(
            curves.flatten().collect_vec(),
        )])
    }
}

#[derive(Debug)]
enum Corner<T: FloatingPoint> {
    Intersection(Point2<T>),
    Curve(NurbsCurve2D<T>),
}

/// Trim a polyline by start and end parameters
fn trim_polyline<T: FloatingPoint>(
    polyline: &NurbsCurve2D<T>,
    start: Option<Point2<T>>,
    end: Option<Point2<T>>,
) -> NurbsCurve2D<T> {
    let dehomogenized_control_points = polyline.dehomogenized_control_points();
    let start = match start {
        Some(p) => p,
        None => dehomogenized_control_points[0],
    };
    let end = match end {
        Some(p) => p,
        None => dehomogenized_control_points[dehomogenized_control_points.len() - 1],
    };
    let pts = vec![start]
        .into_iter()
        .chain(
            dehomogenized_control_points
                .iter()
                .skip(1)
                .take(dehomogenized_control_points.len() - 2)
                .cloned(),
        )
        .chain(vec![end])
        .collect_vec();
    NurbsCurve2D::polyline(&pts, false)
}

/// Find the corner between two compound curves
fn find_corner<T: FloatingPoint>(
    s0: &[CompoundCurve2D<T>],
    s1: &[CompoundCurve2D<T>],
    option: &CurveOffsetOption<T>,
) -> anyhow::Result<Option<Corner<T>>> {
    match (s0.len(), s1.len()) {
        (1, 1) => {
            let last = s0[0].spans().last();
            let head = s1[0].spans().first();
            match (last, head) {
                (Some(last), Some(head)) => corner(last, head, option),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

/// Find the corner between two spans
fn corner<T: FloatingPoint>(
    c0: &NurbsCurve2D<T>,
    c1: &NurbsCurve2D<T>,
    option: &CurveOffsetOption<T>,
) -> anyhow::Result<Option<Corner<T>>> {
    let last_degree = c0.degree();
    let head_degree = c1.degree();
    if last_degree == 1 && last_degree == head_degree {
        // find intersection of the two spans
        let pt0 = c0.dehomogenized_control_points();
        let n0 = pt0.len();
        let pt1 = c1.dehomogenized_control_points();
        let s0 = pt0[n0 - 2..].iter().collect_vec();
        let s1 = pt1[..2].iter().collect_vec();
        let l0 = to_line_helper(s0[0], s0[1]);
        let l1 = to_line_helper(s1[0], s1[1]);
        let it = geo::algorithm::line_intersection::line_intersection(l0, l1);
        match it {
            Some(LineIntersection::SinglePoint {
                intersection,
                is_proper: _,
            }) => {
                let pt = Point2::new(
                    T::from_f64(intersection.x).unwrap(),
                    T::from_f64(intersection.y).unwrap(),
                );
                Ok(Some(Corner::Intersection(pt)))
            }
            _ => {
                // create a corner curve
                match option.corner_type() {
                    CurveOffsetCornerType::None => unreachable!(),
                    CurveOffsetCornerType::Sharp => {
                        let delta = option.distance().abs() * T::from_f64(2.0).unwrap();
                        let it = sharp_corner_intersection([s0[0], s0[1], s1[0], s1[1]], delta)?;
                        let corner =
                            NurbsCurve2D::polyline(&[s0[1].clone(), it, s1[0].clone()], false);
                        Ok(Some(Corner::Curve(corner)))
                    }
                    CurveOffsetCornerType::Round => {
                        let arc = round_corner(s0[0], s0[1], s1[0], *option.distance())?;
                        Ok(Some(Corner::Curve(arc)))
                    }
                    CurveOffsetCornerType::Smooth => {
                        let bezier = smooth_corner(s0[0], s0[1], s1[0], s1[1], *option.distance())?;
                        Ok(Some(Corner::Curve(bezier)))
                    }
                    CurveOffsetCornerType::Chamfer => {
                        let last = pt0.last().ok_or(anyhow::anyhow!("No last point"))?;
                        let head = pt1.first().ok_or(anyhow::anyhow!("No head point"))?;
                        Ok(Some(Corner::Curve(NurbsCurve2D::polyline(
                            &[*last, *head],
                            false,
                        ))))
                    }
                }
            }
        }
    } else {
        // just connect the two spans by corner
        Ok(None)
    }
}

/// Get the parameter at the intersection of the two spans
fn get_parameter_at_intersection<T: FloatingPoint>(polyline: &NurbsCurve2D<T>, t01: T) -> T {
    let knots = polyline.knots();
    let n = polyline.control_points().len();
    let d = knots[n] - knots[n - 1];
    knots[n - 1] + d * t01
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;

    use crate::curve::NurbsCurve2D;

    #[test]
    fn test() {
        let polyline = NurbsCurve2D::polyline(
            &[
                Point2::new(0., 0.),
                Point2::new(1., 0.),
                Point2::new(1., 1.),
                Point2::new(0., 1.),
            ],
            false,
        );

        let knots = polyline.knots();
        println!("{:?}", knots);
        let pt1 = polyline.point_at(knots[1]);
        let pt2 = polyline.point_at(knots[2]);
        let pt3 = polyline.point_at(knots[3]);
        let pt4 = polyline.point_at(knots[4]);
        println!("{:?}", pt1);
        println!("{:?}", pt2);
        println!("{:?}", pt3);
        println!("{:?}", pt4);
    }
}
