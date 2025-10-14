use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{Const, Point3, Vector3};

use crate::{
    curve::nurbs_curve::NurbsCurve, misc::FloatingPoint, morph::Morph,
    surface::nurbs_surface::NurbsSurface,
};

// Implementation for NurbsCurve with adaptive subdivision based on surface normals
impl<T> Morph<T, Const<4>> for NurbsCurve<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat,
{
    type Output = NurbsCurve<T, Const<4>>;

    /// Morphs a curve from the reference surface to the target surface.
    ///
    /// This implementation uses adaptive subdivision based on surface normal changes
    /// starting from Greville abscissae (parameters corresponding to control points).
    /// This ensures the morphed curve properly follows the target surface geometry
    /// while respecting the original curve's parametric structure.
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, Const<4>>,
        target_surface: &NurbsSurface<T, Const<4>>,
    ) -> anyhow::Result<Self::Output> {
        let parameters = self.greville_abscissae()?;

        let pts = if self.degree() == 1 {
            self.dehomogenized_control_points()
        } else {
            parameters.iter().map(|p| self.point_at(*p)).collect_vec()
        };

        let pts = pts
            .into_iter()
            .zip(parameters.into_iter())
            .map(|(pt, t)| {
                let m = super::point_morph::morph_point(&pt, reference_surface, target_surface)?;
                Ok(MorphPoint {
                    point: m.0,
                    normal: m.1,
                    parameter: t,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        todo!()
    }
}

fn subdivide<T: FloatingPoint + ArgminFloat>(
    curve: &NurbsCurve<T, Const<4>>,
    left: &MorphPoint<T>,
    right: &MorphPoint<T>,
    reference_surface: &NurbsSurface<T, Const<4>>,
    target_surface: &NurbsSurface<T, Const<4>>,
) -> anyhow::Result<Vec<MorphPoint<T>>> {
    let mid = (left.parameter + right.parameter) / T::from_f64(2.0).unwrap();
    let pt = curve.point_at(mid);
    let m = super::point_morph::morph_point(&pt, reference_surface, target_surface)?;
    todo!()
}

#[derive(Debug, Clone)]
struct MorphPoint<T: FloatingPoint> {
    point: Point3<T>,
    normal: Vector3<T>,
    parameter: T,
}
