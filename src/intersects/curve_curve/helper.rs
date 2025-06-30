use geo::{BoundingRect, LineIntersection};
use itertools::Itertools;
use nalgebra::Point2;
use num_traits::NumCast;

use crate::misc::FloatingPoint;

/// Convert a points into geo::LineString
pub fn to_line_string_helper<T: FloatingPoint>(points: &[Point2<T>]) -> geo::LineString {
    geo::LineString::new(
        points
            .iter()
            .map(|p| {
                geo::Coord::from([
                    <f64 as NumCast>::from(p.x).unwrap(),
                    <f64 as NumCast>::from(p.y).unwrap(),
                ])
            })
            .collect_vec(),
    )
}

/// Intersection of two line strings
pub struct LineStringIntersection {
    pub point: Point2<f64>,
    pub line_index: (usize, usize),
}

impl LineStringIntersection {
    pub fn point(&self) -> Point2<f64> {
        self.point
    }

    pub fn line_index(&self) -> (usize, usize) {
        self.line_index
    }
}

/// Find the intersection of two line strings
pub fn find_line_string_intersection(
    l0: &geo::LineString,
    l1: &geo::LineString,
) -> anyhow::Result<Vec<LineStringIntersection>> {
    let b0 = l0
        .bounding_rect()
        .ok_or(anyhow::anyhow!("no bounding rect for line: {:?}", l0))?;
    let b1 = l1
        .bounding_rect()
        .ok_or(anyhow::anyhow!("no bounding rect for line: {:?}", l1))?;
    if !geo::Intersects::intersects(&b0, &b1) {
        return Ok(vec![]);
    }

    let mut res = vec![];
    l0.lines().enumerate().for_each(|(i0, l0)| {
        l1.lines().enumerate().for_each(|(i1, l1)| {
            let it = geo::algorithm::line_intersection::line_intersection(l0, l1);
            match it {
                Some(LineIntersection::SinglePoint {
                    intersection,
                    is_proper: _,
                }) => {
                    let p0 = Point2::new(intersection.x, intersection.y);
                    res.push(LineStringIntersection {
                        point: p0,
                        line_index: (i0, i1),
                    });
                }
                _ => {}
            }
        });
    });

    Ok(res)
}
