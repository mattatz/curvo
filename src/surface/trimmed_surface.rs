use argmin::core::ArgminFloat;
use nalgebra::{Point3, Vector3};

use crate::{
    curve::{NurbsCurve2D, NurbsCurve3D},
    misc::FloatingPoint,
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

fn try_project_curve<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface3D<T>,
    curve: &NurbsCurve3D<T>,
    direction: &Vector3<T>,
) -> anyhow::Result<NurbsCurve2D<T>> {
    let weights = curve.weights();
    let pts = curve
        .dehomogenized_control_points()
        .iter()
        .zip(weights.into_iter())
        .map(|(p, w)| {
            surface
                .find_closest_parameter(p)
                .map(|(x, y)| Point3::new(x * w, y * w, w))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    NurbsCurve2D::try_new(curve.degree(), pts, curve.knots().to_vec())
}
