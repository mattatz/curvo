use std::{cmp::Ordering, collections::VecDeque};

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OMatrix,
    OPoint, OVector, U1,
};

use crate::{
    bounding_box::BoundingBox,
    curve::{nurbs_curve::dehomogenize, NurbsCurve, TrimmedCurve},
    misc::{FloatingPoint, Invertible, Transformable},
};

use super::curve_direction::{ends, CurveDirection};

/// The distance below which two ends of spans are connected, as a fraction of the size of the
/// spans all together.
pub const JOINT_DISTANCE: f64 = 1e-4;

/// A struct representing a compound curve.
/// Each span is a `TrimmedCurve` that stores a full NURBS curve plus
/// an active parameter domain. This allows representing sub-intervals
/// of a shared underlying curve without splitting.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    spans: Vec<TrimmedCurve<T, D>>,
}

/// 2D compound curve alias
pub type CompoundCurve2D<T> = CompoundCurve<T, Const<3>>;
/// 3D compound curve alias
pub type CompoundCurve3D<T> = CompoundCurve<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create from a list of NurbsCurve spans (each becomes a full-domain TrimmedCurve).
    pub fn new_unchecked(spans: Vec<NurbsCurve<T, D>>) -> Self {
        Self {
            spans: spans.into_iter().map(TrimmedCurve::from_curve).collect(),
        }
    }

    /// Create from a list of TrimmedCurve spans directly.
    pub fn new_unchecked_trimmed(spans: Vec<TrimmedCurve<T, D>>) -> Self {
        Self { spans }
    }

    /// Create from NurbsCurve spans with aligned knot vectors.
    pub fn new_unchecked_aligned(spans: Vec<NurbsCurve<T, D>>) -> Self {
        let mut knot_offset = T::zero();
        let mut spans = spans;
        spans.iter_mut().for_each(|curve| {
            let start = curve.knots().first();
            curve.knots_mut().iter_mut().for_each(|v| {
                *v = *v - start + knot_offset;
            });
            knot_offset = curve.knots().last();
        });
        Self {
            spans: spans.into_iter().map(TrimmedCurve::from_curve).collect(),
        }
    }

    /// Create from NurbsCurve spans, checking connectivity.
    ///
    /// The spans may come in any order and direction: each is connected to the start or to the
    /// end of those connected so far, reversed if it has to be. Two ends are connected when they
    /// are closer than [`JOINT_DISTANCE`], relative to the size of the spans all together.
    pub fn try_new(spans: Vec<NurbsCurve<T, D>>) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let size = BoundingBox::from_iter(
            spans
                .iter()
                .flat_map(|span| span.control_points())
                .filter_map(dehomogenize),
        )
        .size()
        .norm();
        let epsilon = T::from_f64(JOINT_DISTANCE).unwrap() * size;

        let mut curves = spans.into_iter();
        let first = curves
            .next()
            .ok_or_else(|| anyhow::anyhow!("No span to create a compound curve"))?;
        let mut curves = curves.collect_vec();
        let (mut start, mut end) = ends(&first);
        let mut connected = VecDeque::from([first]);

        while !curves.is_empty() {
            let (index, direction) = curves
                .iter()
                .enumerate()
                .find_map(|(i, curve)| {
                    let these = (start.clone(), end.clone());
                    CurveDirection::between(these, ends(curve), epsilon).map(|d| (i, d))
                })
                .ok_or_else(|| anyhow::anyhow!("No connection found to create a compound curve"))?;
            let next = curves.remove(index);
            match direction {
                CurveDirection::Forward => connected.push_back(next),
                CurveDirection::Backward => connected.push_front(next),
                CurveDirection::Facing => connected.push_back(next.inverse()),
                CurveDirection::Opposite => connected.push_front(next.inverse()),
            }
            (start, end) = (
                ends(&connected[0]).0,
                ends(&connected[connected.len() - 1]).1,
            );
        }
        Ok(Self::new_unchecked_aligned(connected.into()))
    }

    /// Get spans as a slice. Each span is a TrimmedCurve that derefs to NurbsCurve.
    pub fn spans(&self) -> &[TrimmedCurve<T, D>] {
        &self.spans
    }

    /// Get mutable spans.
    pub fn spans_mut(&mut self) -> &mut [TrimmedCurve<T, D>] {
        &mut self.spans
    }

    /// Convert into TrimmedCurves.
    pub fn into_spans(self) -> Vec<TrimmedCurve<T, D>> {
        self.spans
    }

    /// Convert into underlying NurbsCurves (drops domain info).
    pub fn into_nurbs_spans(self) -> Vec<NurbsCurve<T, D>> {
        self.spans.into_iter().map(|tc| tc.into_curve()).collect()
    }

    /// Get the domain of the compound curve.
    pub fn knots_domain(&self) -> (T, T) {
        let knots = self.spans.iter().map(|span| span.domain());
        knots.reduce(|a, b| (a.0.min(b.0), a.1.max(b.1))).unwrap()
    }

    /// Find the index of the span containing the parameter t.
    pub fn find_span_index(&self, t: T) -> usize {
        let index = self.spans.iter().find_position(|span| {
            let (d0, d1) = span.domain();
            (d0..=d1).contains(&t)
        });
        if let Some((index, _)) = index {
            index
        } else if t < self.spans[0].domain().0 {
            0
        } else {
            self.spans.len() - 1
        }
    }

    /// Find the span containing the parameter t.
    pub fn find_span(&self, t: T) -> &NurbsCurve<T, D>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let index = self.find_span_index(t);
        self.spans[index].curve()
    }

    /// Evaluate the curve containing the parameter t at the given parameter t.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{FRAC_PI_2, PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    /// assert_relative_eq!(compound.point_at(0.), Point2::new(1., 0.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(FRAC_PI_2), Point2::new(0., 1.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(PI), Point2::new(-1., 0.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(PI + FRAC_PI_2), Point2::new(0., -1.), epsilon = 1e-5);
    /// assert_relative_eq!(compound.point_at(TAU), Point2::new(1., 0.), epsilon = 1e-5);
    /// ```
    pub fn point_at(&self, t: T) -> OPoint<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let span = self.find_span(t);
        span.point_at(t)
    }

    /// Evaluate the tangent vector of the curve containing the parameter t at the given parameter t.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    /// assert_relative_eq!(compound.tangent_at(0.).normalize(), Vector2::y(), epsilon = 1e-10);
    /// assert_relative_eq!(compound.tangent_at(PI).normalize(), -Vector2::y(), epsilon = 1e-10);
    /// assert_relative_eq!(compound.tangent_at(TAU).normalize(), Vector2::y(), epsilon = 1e-10);
    /// ```
    pub fn tangent_at(&self, t: T) -> OVector<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let span = self.find_span(t);
        span.tangent_at(t)
    }

    /// Check if the curve is closed.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{PI, TAU};
    /// use approx::{assert_relative_eq};
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let circle = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    /// assert!(circle.is_closed(None));
    /// ```
    ///
    /// The curve is closed when its two ends are closer than `epsilon`. Without one, they are to
    /// be connected as its spans are to each other: closer than [`JOINT_DISTANCE`], relative to
    /// the size of the curve.
    pub fn is_closed(&self, epsilon: Option<T>) -> bool
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let (Some(first), Some(last)) = (self.spans.first(), self.spans.last()) else {
            return false;
        };
        let epsilon = epsilon.unwrap_or_else(|| {
            T::from_f64(JOINT_DISTANCE).unwrap() * BoundingBox::from(self).size().norm()
        });
        (first.start_point() - last.end_point()).norm() < epsilon
    }

    /// Returns the total length of the compound curve.
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point2, Vector2};
    /// use std::f64::consts::{PI, TAU};
    /// use approx::{assert_relative_eq};
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::new_unchecked(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]);
    /// let length = compound.try_length().unwrap();
    /// assert_relative_eq!(length, TAU);
    /// ```
    pub fn try_length(&self) -> anyhow::Result<T>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let lengths: anyhow::Result<Vec<T>> =
            self.spans.iter().map(|span| span.try_length()).collect();
        let total = lengths?.iter().fold(T::zero(), |a, b| a + *b);
        Ok(total)
    }

    /// Find the closest point on the curve to a given point
    /// # Example
    /// ```
    /// use nalgebra::{Point2, Vector2};
    /// use curvo::prelude::*;
    /// use std::f64::consts::{PI, TAU};
    /// use approx::assert_relative_eq;
    /// let o = Point2::origin();
    /// let dx = Vector2::x();
    /// let dy = Vector2::y();
    /// let compound = CompoundCurve::try_new(vec![
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
    ///     NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    /// ]).unwrap();
    /// assert_relative_eq!(compound.find_closest_point(&Point2::new(3.0, 0.0)).unwrap(), Point2::new(1., 0.));
    /// ```
    pub fn find_closest_point(
        &self,
        point: &OPoint<T, DimNameDiff<D, U1>>,
    ) -> anyhow::Result<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        T: ArgminFloat,
    {
        let res: anyhow::Result<Vec<_>> = self
            .spans
            .iter()
            .map(|span| span.find_closest_point(point))
            .collect();
        let res = res?;
        let closest = res
            .into_iter()
            .map(|pt| {
                let delta = &pt - point;
                let distance = delta.norm_squared();
                (pt, distance)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        match closest {
            Some(closest) => Ok(closest.0),
            _ => Err(anyhow::anyhow!("Failed to find the closest point")),
        }
    }
}

impl<T: FloatingPoint, D: DimName> FromIterator<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = NurbsCurve<T, D>>>(iter: I) -> Self {
        Self {
            spans: iter.into_iter().map(TrimmedCurve::from_curve).collect(),
        }
    }
}

impl<T: FloatingPoint, D: DimName> From<NurbsCurve<T, D>> for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn from(value: NurbsCurve<T, D>) -> Self {
        Self::new_unchecked(vec![value])
    }
}

impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for CompoundCurve<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.spans
            .iter_mut()
            .for_each(|span| span.transform(transform));
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn invert(&mut self) {
        self.spans.iter_mut().for_each(|span| span.invert());
        self.spans.reverse();
    }
}

#[cfg(feature = "serde")]
impl<T, D: DimName> serde::Serialize for CompoundCurve<T, D>
where
    T: FloatingPoint + serde::Serialize,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>: serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CompoundCurve", 1)?;
        state.serialize_field("spans", &self.spans)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, D: DimName> serde::Deserialize<'de> for CompoundCurve<T, D>
where
    T: FloatingPoint + serde::Deserialize<'de>,
    DefaultAllocator: Allocator<D>,
    <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>: serde::Deserialize<'de>,
{
    fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
    where
        S: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};

        #[derive(Debug)]
        enum Field {
            Spans,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
            where
                S: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("`control_points` or `degree` or `knots`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "spans" => Ok(Field::Spans),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct CompoundCurveVisitor<T, D>(std::marker::PhantomData<(T, D)>);

        impl<T, D> CompoundCurveVisitor<T, D> {
            pub fn new() -> Self {
                CompoundCurveVisitor(std::marker::PhantomData)
            }
        }

        impl<'de, T, D: DimName> Visitor<'de> for CompoundCurveVisitor<T, D>
        where
            T: FloatingPoint + serde::Deserialize<'de>,
            DefaultAllocator: Allocator<D>,
            <DefaultAllocator as nalgebra::allocator::Allocator<D>>::Buffer<T>:
                serde::Deserialize<'de>,
        {
            type Value = CompoundCurve<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct CompoundCurve")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut spans = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Spans => {
                            if spans.is_some() {
                                return Err(de::Error::duplicate_field("spans"));
                            }
                            spans = Some(map.next_value()?);
                        }
                    }
                }
                let spans = spans.ok_or_else(|| de::Error::missing_field("spans"))?;

                Ok(Self::Value { spans })
            }
        }

        const FIELDS: &[&str] = &["spans"];
        deserializer.deserialize_struct(
            "CompoundCurve",
            FIELDS,
            CompoundCurveVisitor::<T, D>::new(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use approx::assert_relative_eq;
    use itertools::Itertools;
    use nalgebra::{Point2, Point3, Vector2, Vector3, U3};
    use std::f64::consts::{FRAC_PI_2, TAU};

    /// Regression: boundaries of a surface extruded from a closed periodic
    /// profile include two closed rim loops; try_new must assemble them without
    /// FP-noise mis-ordering ("No connection found").
    #[test]
    fn try_new_assembles_closed_extrusion_boundary() {
        let pts: Vec<Point3<f64>> = (0..24)
            .map(|i| {
                let t = TAU * i as f64 / 24.0;
                let r = 2.0 + t.cos();
                Point3::new(r * t.cos(), r * t.sin(), 0.0)
            })
            .collect();
        let profile = NurbsCurve3D::interpolate_periodic(&pts, 3, KnotStyle::Centripetal).unwrap();
        let surface = NurbsSurface3D::extrude(&profile, &Vector3::z());
        let curves = surface.try_boundary_curves().unwrap();
        let compound = CompoundCurve::try_new(curves.to_vec());
        assert!(compound.is_ok(), "try_new failed: {:?}", compound.err());
    }

    /// The quarter of the circle of the given radius from `i` to `i + 1` quarter turns.
    fn quarter(i: usize, radius: f64) -> NurbsCurve2D<f64> {
        let turn = FRAC_PI_2 * i as f64;
        NurbsCurve2D::try_arc(
            &Point2::origin(),
            &Vector2::x(),
            &Vector2::y(),
            radius,
            turn,
            turn + FRAC_PI_2,
        )
        .unwrap()
    }

    #[test]
    fn try_new_connects_spans_in_any_order_and_direction_at_any_scale() {
        for scale in [1., 1e-3, 1e3] {
            let q = |i| quarter(i, scale);
            for spans in [
                vec![q(0), q(1), q(2), q(3)],
                vec![q(2), q(0), q(3), q(1)],
                vec![q(1), q(3), q(0), q(2)],
                vec![q(0), q(1).inverse(), q(2), q(3).inverse()],
                vec![q(3).inverse(), q(1), q(0).inverse(), q(2)],
            ] {
                let circle = CompoundCurve::try_new(spans).unwrap();
                assert_eq!(circle.spans().len(), 4);
                // each span starts where the one before it ends, around to the first
                for (a, b) in circle.spans().iter().circular_tuple_windows() {
                    let gap = (a.end_point() - b.start_point()).norm();
                    assert!(gap < 1e-12 * scale, "scaled by {scale}");
                }
                assert!(circle.is_closed(None), "scaled by {scale}");
                assert_relative_eq!(
                    circle.try_length().unwrap(),
                    TAU * scale,
                    epsilon = 1e-8 * scale
                );
            }
        }
    }

    #[test]
    fn try_new_connects_ends_closer_than_a_fraction_of_the_size() {
        let line = |a: f64, b: f64, scale: f64| {
            NurbsCurve2D::polyline(
                &[Point2::new(a * scale, 0.), Point2::new(b * scale, 0.)],
                false,
            )
        };
        for scale in [1., 1e-3, 1e3] {
            // a millionth of the size apart
            let near =
                CompoundCurve::try_new(vec![line(0., 1., scale), line(1. + 1e-6, 2., scale)]);
            assert!(near.is_ok(), "scaled by {scale}");
            assert!(!near.unwrap().is_closed(None), "scaled by {scale}");
            // a twentieth of the size apart
            let apart = CompoundCurve::try_new(vec![line(0., 1., scale), line(1.05, 2., scale)]);
            assert!(apart.is_err(), "scaled by {scale}");
        }
        assert!(CompoundCurve::<f64, U3>::try_new(vec![]).is_err());
    }
}
