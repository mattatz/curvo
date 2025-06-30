use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OMatrix,
    OPoint, OVector, U1,
};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Invertible, Transformable},
};

use super::curve_direction::CurveDirection;

/// A struct representing a compound curve.
#[derive(Clone, Debug, PartialEq)]
pub struct CompoundCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    spans: Vec<NurbsCurve<T, D>>,
}

/// 2D compound curve alias
pub type CompoundCurve2D<T> = CompoundCurve<T, Const<3>>;
/// 3D compound curve alias
pub type CompoundCurve3D<T> = CompoundCurve<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> CompoundCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    /// Create a new compound curve from a list of spans without checking if the spans are connected.
    pub fn new_unchecked(spans: Vec<NurbsCurve<T, D>>) -> Self {
        Self { spans }
    }

    /// Create a new compound curve from a list of spans without checking if the spans are connected.
    /// The knot vectors of the spans are aligned to the first span's knot vector.
    pub fn new_unchecked_aligned(spans: Vec<NurbsCurve<T, D>>) -> Self {
        // Align knot vectors
        // The first knot vector starts at 0, the rest are aligned to the previous knot vector
        let mut knot_offset = T::zero();

        let mut spans = spans;
        spans.iter_mut().for_each(|curve| {
            let start = curve.knots().first();
            curve.knots_mut().iter_mut().for_each(|v| {
                *v = *v - start + knot_offset;
            });
            knot_offset = curve.knots().last();
        });

        Self { spans }
    }

    /// Create a new compound curve from a list of spans.
    /// The spans must be connected.
    pub fn try_new(spans: Vec<NurbsCurve<T, D>>) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        // epsilon for determining the connected points
        let epsilon = T::from_f64(1e-4).unwrap();

        // Ensure the adjacent spans are connected in the forward direction.
        let mut curves = spans.clone();
        let mut connected = vec![curves.remove(0)];

        while !curves.is_empty() {
            let current = connected.len() - 1;
            let last = &connected[current];
            let found = curves.iter().enumerate().find_map(|(i, c)| {
                CurveDirection::new(last, c, epsilon).map(|direction| (i, direction))
            });
            match found {
                Some((index, direction)) => {
                    let next = curves.remove(index);
                    match direction {
                        CurveDirection::Forward => {
                            connected.push(next);
                        }
                        CurveDirection::Backward => {
                            connected.insert(current, next);
                        }
                        CurveDirection::Facing => {
                            connected.push(next.inverse());
                        }
                        CurveDirection::Opposite => {
                            if current == 0 {
                                connected.insert(current, next.inverse());
                            } else {
                                anyhow::bail!("Cannot handle opposite direction");
                            }
                        }
                    }
                }
                None => {
                    anyhow::bail!("No connection found to create a compound curve");
                }
            }
        }

        Ok(Self::new_unchecked_aligned(connected))
    }

    pub fn spans(&self) -> &[NurbsCurve<T, D>] {
        &self.spans
    }

    pub fn spans_mut(&mut self) -> &mut [NurbsCurve<T, D>] {
        &mut self.spans
    }

    /// Convert the compound curve into a vector of spans.
    pub fn into_spans(self) -> Vec<NurbsCurve<T, D>> {
        self.spans
    }

    /// Get the domain of the compound curve
    pub fn knots_domain(&self) -> (T, T) {
        let knots = self.spans.iter().map(|span| span.knots_domain());
        knots.reduce(|a, b| (a.0.min(b.0), a.1.max(b.1))).unwrap()
    }

    /// Find the index of the span containing the parameter t.
    pub fn find_span_index(&self, t: T) -> usize {
        let index = self.spans.iter().find_position(|span| {
            let (d0, d1) = span.knots_domain();
            (d0..=d1).contains(&t)
        });
        if let Some((index, _)) = index {
            index
        } else if t < self.spans[0].knots_domain().0 {
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
        &self.spans[index]
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
    pub fn is_closed(&self, epsilon: Option<T>) -> bool
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let start = self.spans.first().map(|s| s.point_at(s.knots_domain().0));
        let end = self.spans.last().map(|s| s.point_at(s.knots_domain().1));
        let eps = epsilon.unwrap_or(T::default_epsilon() * T::from_usize(10).unwrap());
        match (start, end) {
            (Some(start), Some(end)) => {
                let delta = start - end;
                delta.norm() < eps
            }
            _ => false,
        }
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
        let lengthes: anyhow::Result<Vec<T>> =
            self.spans.iter().map(|span| span.try_length()).collect();
        let total = lengthes?.iter().fold(T::zero(), |a, b| a + *b);
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
            spans: iter.into_iter().collect(),
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
