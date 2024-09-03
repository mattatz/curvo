use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimName, OPoint, Point2};

use crate::prelude::Contains;

use super::{orientation, FloatingPoint, Orientation};

/// A boundary of a polygon curve.
#[derive(Debug, Clone)]
pub struct PolygonBoundary<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    vertices: Vec<OPoint<T, D>>,
}

impl<T: FloatingPoint, D: DimName> PolygonBoundary<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(vertices: Vec<OPoint<T, D>>) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &Vec<OPoint<T, D>> {
        &self.vertices
    }
}

impl<T: FloatingPoint, D: DimName> FromIterator<OPoint<T, D>> for PolygonBoundary<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = OPoint<T, D>>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// Check if a point is contained in a polygon curve.
/// ```
/// use nalgebra::Point2;
/// use curvo::prelude::PolygonBoundary;
/// use curvo::prelude::Contains;
/// let boundary = vec![
///   Point2::new(0., 0.),
///   Point2::new(1., 0.),
///   Point2::new(1., 1.),
///   Point2::new(0., 1.),
/// ];
/// let curve = PolygonBoundary::new(boundary);
/// assert!(curve.contains(&Point2::new(0.5, 0.5), ()).unwrap());
/// assert!(!curve.contains(&Point2::new(0.5, 1.5), ()).unwrap());
/// ```
impl<T: FloatingPoint> Contains<T, Const<2>> for PolygonBoundary<T, Const<2>> {
    type Option = ();

    fn contains(&self, c: &Point2<T>, _option: Self::Option) -> anyhow::Result<bool> {
        let winding_number = self.vertices().iter().circular_tuple_windows().fold(
            0_i32,
            move |winding_number, (p0, p1)| {
                if p0.y <= c.y {
                    if p1.y >= c.y {
                        let o = orientation(p0, p1, c);
                        if o == Orientation::CounterClockwise && p1.y != c.y {
                            return winding_number + 1;
                        }
                    }
                } else if p1.y <= c.y {
                    let o = orientation(p0, p1, c);
                    if o == Orientation::Clockwise {
                        return winding_number - 1;
                    }
                }
                winding_number
            },
        );
        Ok(winding_number != 0)
    }
}
