use geo::LineIntersection;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, Point, Point2, Vector2, U1,
};
use num_traits::NumCast;

use crate::curve::{NurbsCurve2D, NurbsCurve3D};
use crate::offset::CurveOffsetCornerType;
use crate::region::{CompoundCurve, CompoundCurve2D};
use crate::tessellation::tessellation_curve::tessellate_curve_adaptive;
use crate::{curve::NurbsCurve, misc::FloatingPoint, offset::Offset};

#[derive(Debug, Clone, PartialEq)]
pub struct CurveOffsetOption<T> {
    distance: T,
    normal_tolerance: T,
    corner_type: CurveOffsetCornerType,
}

impl<T: FloatingPoint> Default for CurveOffsetOption<T> {
    fn default() -> Self {
        Self {
            distance: T::zero(),
            normal_tolerance: T::from_f64(1e-4).unwrap(),
            corner_type: Default::default(),
        }
    }
}

impl<T> CurveOffsetOption<T> {
    pub fn with_distance(mut self, distance: T) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_normal_tolerance(mut self, tol: T) -> Self {
        self.normal_tolerance = tol;
        self
    }

    pub fn with_corner_type(mut self, ty: CurveOffsetCornerType) -> Self {
        self.corner_type = ty;
        self
    }
}

/// vertex variant
#[derive(Debug, Clone, PartialEq)]
enum Vertex<T: FloatingPoint> {
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
    pub fn point<P: geo::CoordNum>(p: geo::Point<P>) -> Self {
        let x = p.x().to_f64().unwrap();
        let y = p.y().to_f64().unwrap();
        Vertex::Point(Point2::new(
            T::from_f64(x).unwrap(),
            T::from_f64(y).unwrap(),
        ))
    }

    pub fn intersection<P: geo::CoordNum>(p: geo::Point<P>) -> Self {
        let x = p.x().to_f64().unwrap();
        let y = p.y().to_f64().unwrap();
        Vertex::Intersection(Point2::new(
            T::from_f64(x).unwrap(),
            T::from_f64(y).unwrap(),
        ))
    }

    pub fn inner(&self) -> &Point2<T> {
        match self {
            Vertex::Point(p) => p,
            Vertex::Intersection(p) => p,
        }
    }
}

impl<'a, T> Offset<'a, T> for NurbsCurve2D<T>
where
    T: FloatingPoint,
{
    type Output = anyhow::Result<CompoundCurve2D<T>>;
    type Option = CurveOffsetOption<T>;

    /// Offset the NURBS curve by a given distance with a given epsilon
    fn offset(&'a self, option: Self::Option) -> Self::Output {
        let CurveOffsetOption {
            distance,
            normal_tolerance: tol,
            corner_type,
        } = option;

        let tess = tessellate_nurbs_curve(self, tol);
        let offset = tess
            .into_iter()
            .map(|(p, tangent)| {
                let t = tangent.normalize();
                let normal = Vector2::new(t.y, -t.x);
                p + normal * distance
            })
            .collect_vec();

        // find intersections

        if self.degree() == 1 {
            let lines = offset
                .chunks(2)
                .map(|w| {
                    let p0 = &w[0];
                    let p1 = &w[1];
                    to_line_helper(p0, p1)
                })
                .collect_vec();

            let its = lines
                .windows(2)
                .map(|w| {
                    let l0 = &w[0];
                    let l1 = &w[1];
                    geo::algorithm::line_intersection::line_intersection(l0.clone(), l1.clone())
                })
                .collect_vec();

            let n = lines.len();
            let vertices: Vec<Vertex<T>> = lines
                .into_iter()
                .enumerate()
                .flat_map(|(i, line)| {
                    if i == 0 {
                        // head
                        match &its[0] {
                            Some(LineIntersection::SinglePoint {
                                intersection,
                                is_proper: _,
                            }) => {
                                vec![
                                    line.start_point().into(),
                                    Vertex::intersection((*intersection).into()),
                                ]
                            }
                            _ => {
                                vec![line.start_point().into(), line.end_point().into()]
                            }
                        }
                    } else if i == n - 1 {
                        // tail
                        let prev = &its[i - 1];
                        match prev {
                            Some(_) => {
                                vec![line.end_point().into()]
                            }
                            _ => {
                                vec![line.start_point().into(), line.end_point().into()]
                            }
                        }
                    } else {
                        // middle
                        let prev = &its[i - 1];
                        let cur = &its[i];
                        let start = match prev {
                            Some(LineIntersection::SinglePoint {
                                intersection: _,
                                is_proper: _,
                            }) => None,
                            _ => Some(line.start_point().into()),
                        };

                        let end = match cur {
                            Some(LineIntersection::SinglePoint {
                                intersection,
                                is_proper: _,
                            }) => Vertex::intersection((*intersection).into()),
                            _ => line.end_point().into(),
                        };

                        vec![start, Some(end)].into_iter().flatten().collect_vec()
                    }
                })
                .collect_vec();

            return match corner_type {
                CurveOffsetCornerType::None => Ok(Self::polyline(&offset, false).into()),
                // CurveOffsetCornerType::None => Ok(Self::polyline(&vertices.into_iter().map(|v| v.into()).collect_vec(), false).into()),
                CurveOffsetCornerType::Sharp => {
                    // scan to connect vertices
                    let n = vertices.len();
                    let mut cursor = 0;
                    let mut scanned: Vec<Point2<T>> = vec![vertices[0].clone().into()];

                    let delta = distance.abs() * T::from_f64(2.0).unwrap();

                    while cursor < n {
                        if cursor + 3 >= n {
                            let rest: Vec<Point2<T>> = vertices[cursor..]
                                .iter()
                                .map(|v| v.clone().into())
                                .collect_vec();
                            scanned.extend(rest);
                            break;
                        }

                        let v1 = &vertices[cursor + 1];
                        if let Vertex::Intersection(it) = v1 {
                            scanned.push(it.clone());
                            cursor += 1;
                            continue;
                        }

                        // find the intersection of the line v0-v1 & v2-v3
                        let v0 = &vertices[cursor];
                        let d0 = (v1.inner() - v0.inner()).normalize() * delta;
                        let l0 = to_line_helper(v0.inner(), &(v1.inner() + d0));

                        let v2 = &vertices[cursor + 2];
                        let v3 = &vertices[cursor + 3];
                        let d1 = (v3.inner() - v2.inner()).normalize() * delta;
                        let l1 = to_line_helper(&(v2.inner() - d1), v3.inner());

                        let it = geo::algorithm::line_intersection::line_intersection(l0, l1);
                        let it = it
                            .and_then(|it| match it {
                                LineIntersection::SinglePoint {
                                    intersection: p,
                                    is_proper: _,
                                } => Some(p),
                                _ => None,
                            })
                            .ok_or(anyhow::anyhow!("no intersection"))?;
                        scanned.push(Point2::new(
                            T::from_f64(it.x).unwrap(),
                            T::from_f64(it.y).unwrap(),
                        ));
                        cursor += 2;
                    }
                    Ok(Self::polyline(&scanned, false).into())
                }
                CurveOffsetCornerType::Round => {
                    // scan to create rounded corner by arc
                    let mut spans = vec![];
                    let n = vertices.len();
                    let mut cursor = 0;
                    let n = vertices.len();
                    let mut scanned: Vec<Point2<T>> = vec![vertices[0].clone().into()];

                    while cursor < n {
                        if cursor + 3 >= n {
                            let rest: Vec<Point2<T>> = vertices[cursor + 1..]
                                .iter()
                                .map(|v| v.clone().into())
                                .collect_vec();
                            scanned.extend(rest);
                            break;
                        }

                        let v1 = &vertices[cursor + 1];
                        scanned.push(v1.inner().clone());

                        if let Vertex::Intersection(_) = v1 {
                            cursor += 1;
                            continue;
                        }

                        spans.push(Self::polyline(&scanned, false));

                        let v0 = &vertices[cursor];
                        let t = (v1.inner() - v0.inner()).normalize();
                        let n = Vector2::new(t.y, -t.x);
                        let d = n * distance;
                        let center = v1.inner() - d;
                        let v2 = &vertices[cursor + 2];

                        // create arc between v1 and v2
                        let arc = Self::try_arc(&center, &n, &-t, distance, T::zero(), T::frac_pi_2())?;
                        // let arc = Self::try_arc(&center, &-t, &n, distance, T::zero(), T::frac_pi_2())?;
                        spans.push(arc);

                        cursor += 2;
                        scanned = vec![v2.inner().clone()];
                    }

                    if !scanned.is_empty() {
                        spans.push(Self::polyline(&scanned, false));
                    }

                    println!("spans: {:?}", spans.len());

                    Ok(CompoundCurve::new_unchecked(spans))
                }
                CurveOffsetCornerType::Smooth => todo!(),
                CurveOffsetCornerType::Chamfer => Self::try_interpolate(
                    &vertices.into_iter().map(|v| v.into()).collect_vec(),
                    self.degree(),
                )
                .map(|c| c.into()),
            };
        } else {
        };

        todo!()
    }
}

impl<'a, T> Offset<'a, T> for NurbsCurve3D<T>
where
    T: FloatingPoint,
{
    type Output = Self;
    type Option = CurveOffsetOption<T>;

    /// Offset the NURBS curve by a given distance with a given epsilon
    fn offset(&'a self, option: Self::Option) -> Self::Output {
        let CurveOffsetOption {
            distance,
            normal_tolerance: epsilon,
            corner_type,
        } = option;

        let tess = tessellate_nurbs_curve(self, epsilon);
        let offset = tess
            .into_iter()
            .map(|(p, tangent)| {
                let normal = tangent;
                p + normal * distance
            })
            .collect_vec();

        todo!()
    }
}

fn tessellate_nurbs_curve<T: FloatingPoint, D: DimName>(
    curve: &NurbsCurve<T, D>,
    normal_tolerance: T,
) -> Vec<(
    OPoint<T, DimNameDiff<D, U1>>,
    OVector<T, DimNameDiff<D, U1>>,
)>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    if curve.degree() == 1 {
        let pts = curve.dehomogenized_control_points();
        pts.windows(2)
            .map(|w| {
                let p0 = &w[0];
                let p1 = &w[1];
                let tangent = p1 - p0;
                [(p0.clone(), tangent.clone()), (p1.clone(), tangent)]
            })
            .flatten()
            .collect()
    } else {
        let mut rng = rand::rng();
        let (start, end) = curve.knots_domain();
        tessellate_curve_adaptive(curve, start, end, normal_tolerance, &mut rng, &|t, p| {
            (p, curve.tangent_at(t))
        })
    }
}

/// Convert a line segment to a geo::Line
fn to_line_helper<T: FloatingPoint>(p0: &Point2<T>, p1: &Point2<T>) -> geo::Line {
    let x0 = <f64 as NumCast>::from(p0.x).unwrap();
    let y0 = <f64 as NumCast>::from(p0.y).unwrap();
    let x1 = <f64 as NumCast>::from(p1.x).unwrap();
    let y1 = <f64 as NumCast>::from(p1.y).unwrap();
    geo::Line::new(geo::coord! { x: x0, y: y0 }, geo::coord! { x: x1, y: y1 })
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;

    use crate::curve::NurbsCurve2D;

    use super::*;

    #[test]
    fn offset_nurbs_curve() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let polyline = NurbsCurve2D::polyline(&points, false);
        let knots = polyline.knots().to_vec();
        let n = knots.len();
        let knots = knots[1..n - 1].to_vec(); // remove the first and last knot
    }
}
