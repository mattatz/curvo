use std::cmp::Ordering;

use argmin::core::ArgminFloat;
use nalgebra::{ComplexField, Matrix4, Point3, Vector3, U3};

use crate::{
    curve::{NurbsCurve2D, NurbsCurve3D},
    misc::{FloatingPoint, Invertible, Transformable},
    prelude::{BoundingBox, HasIntersection, Intersects},
    region::{CompoundCurve, CompoundCurve2D, CompoundCurve3D},
};

use super::NurbsSurface3D;

/// A trimmed NURBS surface.
/// Base surface & a set of trimming curves in parameter space
#[derive(Debug, Clone)]
pub struct TrimmedSurface<T: FloatingPoint> {
    surface: NurbsSurface3D<T>,
    exterior: Option<CompoundCurve<T, U3>>,
    interiors: Vec<CompoundCurve<T, U3>>,
}

impl<T: FloatingPoint> TrimmedSurface<T> {
    pub fn new(
        surface: NurbsSurface3D<T>,
        exterior: Option<CompoundCurve<T, U3>>,
        interiors: Vec<CompoundCurve<T, U3>>,
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
            exterior: exterior.map(|curve| curve.into()),
            interiors: interiors.into_iter().map(|curve| curve.into()).collect(),
        })
    }

    /// Try to map the trimming curves onto the surface using the closest point
    /// This is a more stable method than `try_projection` but may not be as accurate
    pub fn try_map_closest_point(
        surface: NurbsSurface3D<T>,
        exterior: Option<CompoundCurve3D<T>>,
        interiors: Vec<CompoundCurve3D<T>>,
    ) -> anyhow::Result<Self>
    where
        T: ArgminFloat,
    {
        anyhow::ensure!(
            exterior.is_some() || !interiors.is_empty(),
            "No trimming curves provided"
        );
        let exterior = match exterior {
            Some(curve) => Some(try_map_curve_closest_point(&surface, &curve)?),
            None => None,
        };
        let interiors = interiors
            .iter()
            .map(|curve| try_map_curve_closest_point(&surface, curve))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            surface,
            exterior,
            interiors: interiors.into_iter().collect(),
        })
    }

    pub fn surface(&self) -> &NurbsSurface3D<T> {
        &self.surface
    }

    pub fn surface_mut(&mut self) -> &mut NurbsSurface3D<T> {
        &mut self.surface
    }

    pub fn exterior(&self) -> Option<&CompoundCurve<T, U3>> {
        self.exterior.as_ref()
    }

    pub fn exterior_mut(&mut self) -> Option<&mut CompoundCurve<T, U3>> {
        self.exterior.as_mut()
    }

    pub fn interiors(&self) -> &[CompoundCurve<T, U3>] {
        &self.interiors
    }

    pub fn interiors_mut(&mut self) -> &mut [CompoundCurve<T, U3>] {
        &mut self.interiors
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
    let l0 = ComplexField::abs(b0.size().dot(direction));
    let l1 = ComplexField::abs(b1.size().dot(direction));
    let ray_length = (b0.center() - b1.center()).norm() + l0 + l1;
    let offset = direction * T::one();
    let weights = curve.weights();
    let pts = curve
        .dehomogenized_control_points()
        .iter()
        .zip(weights.into_iter())
        .map(|(p, w)| {
            let ray =
                NurbsCurve3D::polyline(&[p - offset, p + offset + direction * ray_length], true);
            let closest = surface
                .find_intersection(&ray, None)?
                .into_iter()
                .map(|it| {
                    let pt = it.a().0;
                    let dist = (p - pt).norm_squared();
                    (dist, it)
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let closest = closest
                .ok_or_else(|| anyhow::anyhow!("No intersection found in try project curve"))?
                .1;
            let uv = closest.a().1;
            Ok(Point3::new(uv.0 * w, uv.1 * w, w))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    NurbsCurve2D::try_new(curve.degree(), pts, curve.knots().to_vec())
}

/// Try to map a 3D curve onto a 3D surface to get a 2D curve in parameter space using the closest point
fn try_map_curve_closest_point<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface3D<T>,
    curve: &CompoundCurve3D<T>,
) -> anyhow::Result<CompoundCurve2D<T>> {
    let spans = curve
        .spans()
        .iter()
        .map(|curve| {
            let weights = curve.weights();
            let pts = curve
                .dehomogenized_control_points()
                .iter()
                .zip(weights.into_iter())
                .map(|(p, w)| {
                    let uv = surface.find_closest_parameter(p)?;
                    Ok(Point3::new(uv.0 * w, uv.1 * w, w))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            NurbsCurve2D::try_new(curve.degree(), pts, curve.knots().to_vec())
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(CompoundCurve2D::new_unchecked(spans))
}

impl<T: FloatingPoint> From<NurbsSurface3D<T>> for TrimmedSurface<T> {
    fn from(value: NurbsSurface3D<T>) -> Self {
        Self::new(value, None, vec![])
    }
}

/// Enable to transform a Trimmed surface by a given DxD matrix
impl<'a, T: FloatingPoint> Transformable<&'a Matrix4<T>> for TrimmedSurface<T> {
    fn transform(&mut self, transform: &'a Matrix4<T>) {
        self.surface.transform(transform);
    }
}

impl<T: FloatingPoint> Invertible for TrimmedSurface<T> {
    /// Reverse the direction of the surface
    fn invert(&mut self) {
        self.surface.invert();
        if let Some(curve) = self.exterior.as_mut() {
            curve.invert();
        }
        self.interiors.iter_mut().for_each(|curve| curve.invert());
    }
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

                impl Visitor<'_> for FieldVisitor {
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
