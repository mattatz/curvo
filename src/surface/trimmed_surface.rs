use std::{cmp::Ordering, f32::EPSILON};

use argmin::core::ArgminFloat;
use nalgebra::{Point3, Vector3, U3};

use crate::{
    curve::{NurbsCurve2D, NurbsCurve3D},
    misc::FloatingPoint,
    prelude::{BoundingBox, HasIntersection, Intersects},
};

use super::NurbsSurface3D;

/// A trimmed NURBS surface.
/// Base surface & a set of trimming curves in parameter space
#[derive(Debug, Clone)]
pub struct TrimmedSurface<T: FloatingPoint> {
    surface: NurbsSurface3D<T>,
    exterior: Option<NurbsCurve2D<T>>,
    interiors: Vec<NurbsCurve2D<T>>,
}

impl<T: FloatingPoint> TrimmedSurface<T> {
    pub fn new(
        surface: NurbsSurface3D<T>,
        exterior: Option<NurbsCurve2D<T>>,
        interiors: Vec<NurbsCurve2D<T>>,
    ) -> Self {
        Self {
            surface,
            exterior,
            interiors,
        }
    }

    /// Try to project the trimming curves onto the surface
    /// Returns an error if the projection fails
    pub fn try_projection(
        surface: NurbsSurface3D<T>,
        direction: Vector3<T>,
        exterior: Option<NurbsCurve3D<T>>,
        interiors: Vec<NurbsCurve3D<T>>,
    ) -> anyhow::Result<Self>
    where
        T: ArgminFloat,
    {
        anyhow::ensure!(
            exterior.is_some() || !interiors.is_empty(),
            "No trimming curves provided"
        );
        let exterior = match exterior {
            Some(curve) => Some(try_project_curve(&surface, &curve, &direction)?),
            None => None,
        };
        let interiors = interiors
            .iter()
            .map(|curve| try_project_curve(&surface, curve, &direction))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            surface,
            exterior,
            interiors,
        })
    }

    pub fn surface(&self) -> &NurbsSurface3D<T> {
        &self.surface
    }

    pub fn exterior(&self) -> Option<&NurbsCurve2D<T>> {
        self.exterior.as_ref()
    }

    pub fn interiors(&self) -> &[NurbsCurve2D<T>] {
        &self.interiors
    }
}

/// Try to project a 3D curve onto a 3D surface to get a 2D curve in parameter space
fn try_project_curve<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface3D<T>,
    curve: &NurbsCurve3D<T>,
    direction: &Vector3<T>,
) -> anyhow::Result<NurbsCurve2D<T>> {
    let b0: BoundingBox<T, U3> = surface.into();
    let b1: BoundingBox<T, U3> = curve.into();
    let ray_length = (b0.center() - b1.center()).norm() + b0.size().norm() + b1.size().norm();
    let offset = -direction * T::epsilon();
    let weights = curve.weights();
    let pts = curve
        .dehomogenized_control_points()
        .iter()
        .zip(weights.into_iter())
        .map(|(p, w)| {
            let ray = NurbsCurve3D::polyline(&[p + offset, p + direction * ray_length], true);
            let closest = surface
                .find_intersections(&ray, None)?
                .into_iter()
                .map(|it| {
                    let pt = it.a().0;
                    let dist = (p - pt).norm_squared();
                    (dist, it)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let closest = closest
                .ok_or_else(|| anyhow::anyhow!("No intersection found"))?
                .1;
            let uv = closest.a().1;
            Ok(Point3::new(uv.0 * w, uv.1 * w, w))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    NurbsCurve2D::try_new(curve.degree(), pts, curve.knots().to_vec())
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for TrimmedSurface<T>
where
    T: FloatingPoint + serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TrimmedSurface", 3)?;
        state.serialize_field("surface", &self.surface)?;
        state.serialize_field("exterior", &self.exterior)?;
        state.serialize_field("interiors", &self.interiors)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T> serde::Deserialize<'de> for TrimmedSurface<T>
where
    T: FloatingPoint + serde::Deserialize<'de>,
{
    fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
    where
        S: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};

        #[derive(Debug)]
        enum Field {
            Surface,
            Exterior,
            Interiors,
        }

        impl<'de> serde::Deserialize<'de> for Field {
            fn deserialize<S>(deserializer: S) -> Result<Self, S::Error>
            where
                S: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("`surface` or `exterior` or `interiors`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "surface" => Ok(Field::Surface),
                            "exterior" => Ok(Field::Exterior),
                            "interiors" => Ok(Field::Interiors),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TrimmedSurfaceVisitor<T>(std::marker::PhantomData<T>);

        impl<T> TrimmedSurfaceVisitor<T> {
            pub fn new() -> Self {
                TrimmedSurfaceVisitor(std::marker::PhantomData)
            }
        }

        impl<'de, T> Visitor<'de> for TrimmedSurfaceVisitor<T>
        where
            T: FloatingPoint + serde::Deserialize<'de>,
        {
            type Value = TrimmedSurface<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct TrimmedSurface")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut surface = None;
                let mut exterior = None;
                let mut interiors = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Surface => {
                            if surface.is_some() {
                                return Err(de::Error::duplicate_field("surface"));
                            }
                            surface = Some(map.next_value()?);
                        }
                        Field::Exterior => {
                            if exterior.is_some() {
                                return Err(de::Error::duplicate_field("exterior"));
                            }
                            exterior = map.next_value().ok();
                        }
                        Field::Interiors => {
                            if interiors.is_some() {
                                return Err(de::Error::duplicate_field("interiors"));
                            }
                            interiors = Some(map.next_value()?);
                        }
                    }
                }

                Ok(Self::Value {
                    surface: surface.ok_or_else(|| de::Error::missing_field("surface"))?,
                    exterior,
                    interiors: interiors.ok_or_else(|| de::Error::missing_field("interiors"))?,
                })
            }
        }

        const FIELDS: &[&str] = &["surface", "exterior", "interiors"];
        deserializer.deserialize_struct("TrimmedSurface", FIELDS, TrimmedSurfaceVisitor::<T>::new())
    }
}
