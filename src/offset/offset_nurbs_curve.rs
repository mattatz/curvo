use std::marker::PhantomData;

use geo::LineIntersection;
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{
    DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, Point2, Vector2, U1,
};
use num_traits::NumCast;

use crate::curve::NurbsCurve2D;
use crate::offset::CurveOffsetCornerType;
use crate::region::{CompoundCurve, CompoundCurve2D};
use crate::tessellation::tessellation_curve::tessellate_curve_adaptive;
use crate::{curve::NurbsCurve, misc::FloatingPoint, offset::Offset};

/// Offset option for NURBS curves
#[derive(Debug, Clone, PartialEq)]
pub struct CurveOffsetOption<T> {
    /// Offset distance
    distance: T,
    /// Normal tolerance for tessellation
    normal_tolerance: T,
    /// Knot tolerance for reducing knots
    knot_tolerance: T,
    /// Corner type
    corner_type: CurveOffsetCornerType,
}

impl<T: FloatingPoint> Default for CurveOffsetOption<T> {
    fn default() -> Self {
        Self {
            distance: T::zero(),
            normal_tolerance: T::from_f64(1e-4).unwrap(),
            knot_tolerance: T::from_f64(1e-4).unwrap(),
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

    pub fn with_knot_tolerance(mut self, tol: T) -> Self {
        self.knot_tolerance = tol;
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
    type Output = anyhow::Result<Vec<CompoundCurve2D<T>>>;
    type Option = CurveOffsetOption<T>;

    /// Offset the NURBS curve by a given distance with a given epsilon
    fn offset(&'a self, option: Self::Option) -> Self::Output {
        let CurveOffsetOption {
            distance,
            normal_tolerance: norm_tol,
            knot_tolerance: knot_tol,
            corner_type,
        } = option;

        let is_closed = self.is_closed();

        let offset = |point: &Point2<T>, t: &Vector2<T>| {
            let normal = Vector2::new(t.y, -t.x);
            let d = normal * distance;
            point + d
        };

        if self.degree() == 1 {
            let pts = self.dehomogenized_control_points();
            let p_segments = pts
                .windows(2)
                .map(|w| {
                    let p0 = &w[0];
                    let p1 = &w[1];
                    let tangent = p1 - p0;
                    let t = tangent.normalize();
                    let start = offset(p0, &t);
                    let end = offset(p1, &t);
                    PointSegment::new(start, end)
                })
                .collect_vec();

            if matches!(corner_type, CurveOffsetCornerType::None) {
                return Ok(p_segments
                    .into_iter()
                    .map(|s| NurbsCurve2D::polyline(&[s.start, s.end], false).into())
                    .collect_vec());
            }

            let n = if is_closed {
                p_segments.len() + 1
            } else {
                p_segments.len()
            };
            let intersections = p_segments
                .iter()
                .cycle()
                .take(n)
                .collect_vec()
                .windows(2)
                .map(|w| {
                    if let Some(p) = w[0].intersects(w[1]) {
                        Vertex::Intersection(p)
                    } else {
                        Vertex::Point(w[0].end)
                    }
                })
                .collect_vec();

            let slen = p_segments.len();
            let segments = p_segments
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    let start = if i == 0 {
                        if is_closed {
                            match intersections[intersections.len() - 1] {
                                Vertex::Intersection(p) => Vertex::Intersection(p),
                                Vertex::Point(_) => Vertex::Point(s.start),
                            }
                        } else {
                            Vertex::Point(s.start)
                        }
                    } else {
                        let prev = intersections[i - 1].clone();
                        match prev {
                            Vertex::Intersection(p) => Vertex::Intersection(p),
                            Vertex::Point(_) => Vertex::Point(s.start),
                        }
                    };
                    let end = if i != slen - 1 || is_closed {
                        intersections[i].clone()
                    } else {
                        Vertex::Point(s.end)
                    };

                    s.to_vertex_segment(start, end)
                })
                .collect_vec();

            let spans = match corner_type {
                CurveOffsetCornerType::None => {
                    unreachable!()
                }
                CurveOffsetCornerType::Sharp => {
                    let delta = distance.abs() * T::from_f64(2.0).unwrap();

                    let n = segments.len();

                    let pts = segments
                        .iter()
                        .enumerate()
                        .map(|(i, s)| match s.start {
                            Vertex::Point(p) => {
                                let prev = if i != 0 || is_closed {
                                    Some(&segments[(i + n - 1) % n])
                                } else {
                                    None
                                };
                                match prev {
                                    Some(prev) => {
                                        let v0 = prev.start.clone();
                                        let v1 = prev.end.clone();
                                        let v2 = s.start.clone();
                                        let v3 = s.end.clone();
                                        let it =
                                            find_corner_intersection([&v0, &v1, &v2, &v3], delta)?;
                                        Ok(it)
                                    }
                                    _ => Ok(p),
                                }
                            }
                            Vertex::Intersection(p) => Ok(p),
                        })
                        .collect::<anyhow::Result<Vec<_>>>()?;

                    let last = segments.last();
                    let last = if let Some(last) = last {
                        match &last.end {
                            Vertex::Point(p) => {
                                if is_closed {
                                    Some(pts[0])
                                } else {
                                    Some(*p)
                                }
                            }
                            Vertex::Intersection(opoint) => Some(*opoint),
                        }
                    } else {
                        None
                    };

                    let pts = if let Some(last) = last {
                        pts.into_iter().chain(vec![last]).collect_vec()
                    } else {
                        pts
                    };

                    return Ok(vec![NurbsCurve2D::polyline(&pts, false).into()]);
                }
                CurveOffsetCornerType::Round => {
                    try_connect(&segments, |cursor| {
                        let cur = &segments[cursor];
                        if cursor == segments.len() - 1 && !is_closed {
                            return Ok(None);
                        }
                        let next = &segments[(cursor + 1) % segments.len()];

                        let v0 = &cur.start;
                        let v1 = &cur.end;
                        let t = (v1.inner() - v0.inner()).normalize();
                        let sign = distance.signum();
                        let n = Vector2::new(t.y, -t.x);
                        let d = n * distance;
                        let center = v1.inner() - d;
                        let v2 = &next.start;

                        let d0 = v1.inner() - center;
                        let d1 = v2.inner() - center;
                        let angle = d0.angle(&d1);
                        let angle = angle.abs();

                        // create arc between v1 and v2
                        let arc = Self::try_arc(
                            &center,
                            &(n * sign),
                            &t,
                            distance.abs(),
                            T::zero(),
                            angle,
                        )?;
                        Ok(Some(arc))
                    })?
                }
                CurveOffsetCornerType::Smooth => {
                    let frac_2_3 = T::from_f64(2.0 / 3.0).unwrap();
                    let d = distance.abs() * frac_2_3;

                    try_connect(&segments, |cursor| {
                        let cur = &segments[cursor];
                        if cursor == segments.len() - 1 && !is_closed {
                            return Ok(None);
                        }
                        let next = &segments[(cursor + 1) % segments.len()];

                        let v0 = &cur.start;
                        let v1 = &cur.end;
                        let v2 = &next.start;
                        let v3 = &next.end;

                        let d10 = if v1.inner() == v0.inner() {
                            Vector2::zeros()
                        } else {
                            (v1.inner() - v0.inner()).normalize() * d
                        };
                        let d32 = if v3.inner() == v2.inner() {
                            Vector2::zeros()
                        } else {
                            (v3.inner() - v2.inner()).normalize() * d
                        };

                        // create arc between v1 and v2
                        let bezier = NurbsCurve2D::bezier(&[
                            *v1.inner(),
                            v1.inner() + d10,
                            v2.inner() - d32,
                            *v2.inner(),
                        ]);
                        Ok(Some(bezier))
                    })?
                }
                CurveOffsetCornerType::Chamfer => try_connect(&segments, |cursor| {
                    let cur = &segments[cursor];
                    if cursor == segments.len() - 1 && !is_closed {
                        return Ok(None);
                    }
                    let next = &segments[(cursor + 1) % segments.len()];
                    let v1 = cur.end.clone();
                    let v2 = next.start.clone();
                    Ok(Some(NurbsCurve2D::polyline(&[v1.into(), v2.into()], false)))
                })?,
            };

            // connect polylines
            let mut res = vec![];
            let mut cursor = None;

            let connect = |i: usize, j: usize| {
                let pts = spans[i..j]
                    .iter()
                    .flat_map(|c| c.dehomogenized_control_points())
                    .dedup()
                    .collect_vec();
                NurbsCurve2D::polyline(&pts, false)
            };

            let n = spans.len();
            for i in 0..n {
                if spans[i].degree() != 1 {
                    if let Some(cursor) = cursor {
                        let m = i.saturating_sub(cursor);
                        if m > 0 {
                            let polyline = connect(cursor, i);
                            res.push(polyline);
                        }
                    }
                    res.push(spans[i].clone());
                    cursor = None;
                } else if cursor.is_none() {
                    cursor = Some(i);
                }
            }

            if let Some(cursor) = cursor {
                if n - cursor > 0 {
                    let polyline = connect(cursor, n);
                    res.push(polyline);
                }
            }

            Ok(vec![CompoundCurve::new_unchecked_aligned(res)])
        } else {
            let tess = tessellate_nurbs_curve(self, norm_tol)
                .into_iter()
                .map(|(p, t)| {
                    let t = t.normalize();
                    offset(&p, &t)
                })
                .collect_vec();
            let mut res = Self::try_interpolate(&tess, self.degree())?;
            res.try_reduce_knots(Some(knot_tol))?;
            Ok(vec![res.into()])
        }
    }
}

#[derive(Debug, Clone)]
struct Segment<T, P> {
    start: P,
    end: P,
    line: geo::Line,
    _marker: PhantomData<T>,
}

type PointSegment<T> = Segment<T, Point2<T>>;
type VertexSegment<T> = Segment<T, Vertex<T>>;

impl<T: FloatingPoint> PointSegment<T> {
    fn new(start: Point2<T>, end: Point2<T>) -> Self {
        let line = to_line_helper(&start, &end);
        Self {
            start,
            end,
            line,
            _marker: PhantomData,
        }
    }

    /// Convert a point segment to a vertex segment
    fn to_vertex_segment(&self, start: Vertex<T>, end: Vertex<T>) -> VertexSegment<T> {
        VertexSegment {
            start,
            end,
            line: self.line,
            _marker: PhantomData,
        }
    }
}

impl<T: FloatingPoint, P> Segment<T, P> {
    fn intersects(&self, other: &Segment<T, P>) -> Option<Point2<T>> {
        let l0 = self.line;
        let l1 = other.line;
        let it = geo::algorithm::line_intersection::line_intersection(l0, l1);
        it.and_then(|it| match it {
            LineIntersection::SinglePoint {
                intersection: p,
                is_proper: _,
            } => Some(Point2::new(
                T::from_f64(p.x).unwrap(),
                T::from_f64(p.y).unwrap(),
            )),
            _ => None,
        })
    }
}

/// try to connect segments by a given corner function
fn try_connect<T: FloatingPoint, F>(
    segments: &[VertexSegment<T>],
    corner: F,
) -> anyhow::Result<Vec<NurbsCurve2D<T>>>
where
    F: Fn(usize) -> anyhow::Result<Option<NurbsCurve2D<T>>>,
{
    let spans = segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let polyline =
                NurbsCurve2D::polyline(&[s.start.clone().into(), s.end.clone().into()], false);
            match s.end {
                Vertex::Point(_) => {
                    let c = corner(i)?;
                    match c {
                        Some(c) => Ok(vec![polyline, c]),
                        None => Ok(vec![polyline]),
                    }
                }
                Vertex::Intersection(_) => Ok(vec![polyline]),
            }
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(spans.into_iter().flatten().collect_vec())
}

/// find the intersection of the line v0-v1 & v2-v3
fn find_corner_intersection<T: FloatingPoint>(
    vertices: [&Vertex<T>; 4],
    delta: T,
) -> anyhow::Result<Point2<T>> {
    let v0 = vertices[0];
    let v1 = vertices[1];
    let v2 = vertices[2];
    let v3 = vertices[3];

    let d0 = (v1.inner() - v0.inner()).normalize() * delta;
    let l0 = to_line_helper(v0.inner(), &(v1.inner() + d0));
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

    Ok(Point2::new(
        T::from_f64(it.x).unwrap(),
        T::from_f64(it.y).unwrap(),
    ))
}

/// tessellate the NURBS curve & return the points and tangent vectors
#[allow(clippy::type_complexity)]
fn tessellate_nurbs_curve<T, D>(
    curve: &NurbsCurve<T, D>,
    normal_tolerance: T,
) -> Vec<(
    OPoint<T, DimNameDiff<D, U1>>,
    OVector<T, DimNameDiff<D, U1>>,
)>
where
    T: FloatingPoint,
    D: DimName,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let mut rng = rand::rng();
    let (start, end) = curve.knots_domain();
    tessellate_curve_adaptive(curve, start, end, normal_tolerance, &mut rng, &|t, p| {
        (p, curve.tangent_at(t))
    })
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

    use crate::{
        curve::NurbsCurve2D,
        offset::{CurveOffsetCornerType, CurveOffsetOption, Offset},
    };

    #[test]
    fn offset_nurbs_curve() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let polyline = NurbsCurve2D::polyline(&points, false);
        let option = CurveOffsetOption {
            distance: 0.2,
            corner_type: CurveOffsetCornerType::Sharp,
            ..Default::default()
        };
        let res = polyline.offset(option).unwrap();
        assert_eq!(res.len(), 1);
        let curve = &res[0];
        let spans = curve.spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        let pts = span.dehomogenized_control_points();
        assert_eq!(pts.len(), 4);
        assert_eq!(
            pts,
            vec![
                Point2::new(0.0, -0.2),
                Point2::new(1.2, -0.2),
                Point2::new(1.2, 1.2),
                Point2::new(0.0, 1.2),
            ]
        );
    }

    #[test]
    fn offset_closed_rectangle() {
        let points = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
            Point2::new(0.0, 0.0),
        ];
        let polyline = NurbsCurve2D::polyline(&points, false);
        let option = CurveOffsetOption {
            distance: 0.2,
            corner_type: CurveOffsetCornerType::Sharp,
            ..Default::default()
        };

        // positive offset
        let res = polyline.offset(option.clone()).unwrap();
        assert_eq!(res.len(), 1);
        let curve = &res[0];
        let spans = curve.spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        let pts = span.dehomogenized_control_points();
        assert_eq!(pts.len(), 5);
        assert_eq!(
            pts,
            vec![
                Point2::new(-0.2, -0.2),
                Point2::new(1.2, -0.2),
                Point2::new(1.2, 1.2),
                Point2::new(-0.2, 1.2),
                Point2::new(-0.2, -0.2),
            ]
        );

        // negative offset
        let option = option.clone().with_distance(-0.2);
        let res = polyline.offset(option).unwrap();
        assert_eq!(res.len(), 1);
        let curve = &res[0];
        let spans = curve.spans();
        assert_eq!(spans.len(), 1);
        let span = &spans[0];
        let pts = span.dehomogenized_control_points();
        assert_eq!(pts.len(), 5);
        assert_eq!(
            pts,
            vec![
                Point2::new(0.2, 0.2),
                Point2::new(0.8, 0.2),
                Point2::new(0.8, 0.8),
                Point2::new(0.2, 0.8),
                Point2::new(0.2, 0.2),
            ]
        );
    }
}
