use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{Const, Point3, Vector3};

use crate::{
    curve::nurbs_curve::NurbsCurve, interpolation::Interpolation, misc::FloatingPoint,
    morph::Morph, surface::nurbs_surface::NurbsSurface,
};

// Implementation for NurbsCurve with adaptive subdivision based on surface normals
impl<T> Morph<T, Const<4>> for NurbsCurve<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat,
{
    /// Normal tolerance for adaptive subdivision
    type Option = Option<T>;

    /// The output type after morphing
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
        tolerance: Self::Option,
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
                let m =
                    super::point_morph::morph_point(&pt, reference_surface, target_surface, None)?;
                Ok(MorphPoint {
                    point: m.0,
                    normal: m.1,
                    uv: m.2,
                    parameter: t,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let tolerance = tolerance.unwrap_or(T::from_f64(1e-3).unwrap());
        let morphed = pts
            .windows(2)
            .map(|window| {
                adaptive_subdivide(
                    self,
                    &window[0],
                    &window[1],
                    reference_surface,
                    target_surface,
                    tolerance,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let head = morphed
            .first()
            .and_then(|x| x.first().map(|x| x.point))
            .ok_or(anyhow::anyhow!("Failed to get the first point"))?;

        let pts = morphed.into_iter().fold(vec![head], |acc, x| {
            let tail = x.into_iter().skip(1).map(|x| x.point).collect_vec();
            [acc, tail].concat()
        });

        // return Ok(NurbsCurve::polyline(&pts, false));

        let degree = self
            .degree()
            .max(target_surface.u_degree())
            .max(target_surface.v_degree());
        NurbsCurve::interpolate(&pts, degree)
    }
}

/// Adaptive subdivide the curve using the surface normal
fn adaptive_subdivide<T: FloatingPoint + ArgminFloat>(
    curve: &NurbsCurve<T, Const<4>>,
    left: &MorphPoint<T>,
    right: &MorphPoint<T>,
    reference_surface: &NurbsSurface<T, Const<4>>,
    target_surface: &NurbsSurface<T, Const<4>>,
    normal_tolerance: T,
) -> anyhow::Result<Vec<MorphPoint<T>>> {
    let mid = (left.parameter + right.parameter) / T::from_f64(2.0).unwrap();

    let dn = (left.normal - right.normal).norm_squared();
    if dn <= normal_tolerance {
        return Ok(vec![left.clone(), right.clone()]);
    }

    let pt = curve.point_at(mid);
    let m = super::point_morph::morph_point(&pt, reference_surface, target_surface, Some(left.uv))?;
    let m = MorphPoint {
        point: m.0,
        normal: m.1,
        uv: m.2,
        parameter: mid,
    };
    let lm = (left.normal - m.normal).norm_squared();
    let rm = (right.normal - m.normal).norm_squared();
    if lm > normal_tolerance || rm > normal_tolerance {
        let mut left = adaptive_subdivide(
            curve,
            left,
            &m,
            reference_surface,
            target_surface,
            normal_tolerance,
        )?;
        left.pop();
        let right = adaptive_subdivide(
            curve,
            &m,
            right,
            reference_surface,
            target_surface,
            normal_tolerance,
        )?;
        Ok([left, right].concat())
    } else {
        Ok(vec![left.clone(), right.clone()])
    }
}

#[derive(Debug, Clone)]
struct MorphPoint<T: FloatingPoint> {
    point: Point3<T>,
    normal: Vector3<T>,
    uv: (T, T),
    parameter: T,
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    use crate::{curve::NurbsCurve3D, misc::EndPoints};

    use super::*;

    #[test]
    fn test_morph_curve_plane_to_plane() {
        let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());
        let target_surface =
            NurbsSurface::plane(Point3::new(0.0, 0.0, 1.0), Vector3::y(), Vector3::z());
        let curve = NurbsCurve3D::polyline(
            &vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            false,
        );

        let morphed = curve.morph(&ref_surface, &target_surface, None).unwrap();
        let (start, end) = morphed.end_points();
        assert_relative_eq!(start, Point3::new(0.0, 0.0, 1.0), epsilon = 1e-4);
        assert_relative_eq!(end, Point3::new(0.0, 1.0, 1.0), epsilon = 1e-4);
    }
}
