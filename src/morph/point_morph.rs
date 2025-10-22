use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint, Point3, Vector3};

use crate::{
    misc::{FloatingPoint, Plane},
    morph::Morph,
    surface::nurbs_surface::NurbsSurface,
};

// Implementation for Point3
impl<T> Morph<T, Const<4>> for OPoint<T, Const<3>>
where
    T: FloatingPoint + ArgminFloat,
{
    type Option = ();
    type Output = OPoint<T, Const<3>>;

    /// Morphs a point from the reference surface to the target surface.
    /// # Example
    ///
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    ///
    /// // Create surfaces
    /// let ref_surface = NurbsSurface3D::<f64>::plane(
    ///     Point3::origin(),
    ///     Vector3::x(),
    ///     Vector3::y(),
    /// );
    ///
    /// let target_surface = NurbsSurface3D::<f64>::plane(
    ///     Point3::new(0.0, 0.0, 1.0),
    ///     Vector3::x(),
    ///     Vector3::y(),
    /// );
    ///
    /// // Morph a point
    /// let point = Point3::new(0.5, 0.5, 0.0);
    /// let morphed: Point3<f64> = point.morph(&ref_surface, &target_surface, ()).unwrap();
    /// assert_relative_eq!(morphed.z, 1.0, epsilon = 1e-6);
    /// ```
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, Const<4>>,
        target_surface: &NurbsSurface<T, Const<4>>,
        _option: Self::Option,
    ) -> anyhow::Result<Self::Output> {
        let (uv, uv2) = remap_closest_uv(self, reference_surface, target_surface, None)?;
        let r = reference_surface.point_normal_at(uv.0, uv.1);
        let plane = Plane::new(r.1, r.0.coords.dot(&r.1));
        let distance_plane = plane.signed_distance(&self);
        let t = target_surface.point_normal_at(uv2.0, uv2.1);
        let mapped = t.0 + t.1.normalize() * distance_plane;
        Ok(mapped)
    }
}

/// Remap the UV parameter from the reference surface to the target surface
pub fn remap_closest_uv<T: FloatingPoint + ArgminFloat>(
    point: &Point3<T>,
    reference_surface: &NurbsSurface<T, Const<4>>,
    target_surface: &NurbsSurface<T, Const<4>>,
    hint: Option<(T, T)>,
) -> anyhow::Result<((T, T), (T, T))> {
    // Find the closest UV parameter on the reference surface
    let uv = reference_surface.find_closest_parameter(point, hint)?;
    let ((u_min, u_max), (v_min, v_max)) = reference_surface.knots_domain();
    let ru = (uv.0 - u_min) / (u_max - u_min);
    let rv = (uv.1 - v_min) / (v_max - v_min);

    // Remap the UV parameter to the target surface
    let ((u_min, u_max), (v_min, v_max)) = target_surface.knots_domain();
    let u2 = u_min + ru * (u_max - u_min);
    let v2 = v_min + rv * (v_max - v_min);

    Ok((uv, (u2, v2)))
}

#[allow(clippy::type_complexity)]
pub(crate) fn morph_point<T: FloatingPoint + ArgminFloat>(
    point: &Point3<T>,
    reference_surface: &NurbsSurface<T, Const<4>>,
    target_surface: &NurbsSurface<T, Const<4>>,
    hint: Option<(T, T)>,
) -> anyhow::Result<(Point3<T>, Vector3<T>, (T, T))> {
    let (uv, uv2) = remap_closest_uv(point, reference_surface, target_surface, hint)?;

    // Evaluate the target surface at the same UV parameter
    let (t, n) = target_surface.point_normal_at(uv2.0, uv2.1);
    Ok((t.into(), n, uv))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    use super::*;

    #[test]
    fn test_morph_point_plane_to_plane() {
        // Create a reference plane at z=0
        let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

        // Create a target plane at z=1
        let target_surface =
            NurbsSurface::plane(Point3::new(0.0, 0.0, 1.0), Vector3::x(), Vector3::y());

        // Morph a point from the reference surface to the target surface
        let point = Point3::new(0.5, 0.5, 0.0);
        let morphed = point.morph(&ref_surface, &target_surface, ()).unwrap();

        // The morphed point should maintain x and y, but have z=1
        let epsilon = 1e-4;
        assert_relative_eq!(morphed.x, 0.5, epsilon = epsilon);
        assert_relative_eq!(morphed.y, 0.5, epsilon = epsilon);
        assert_relative_eq!(morphed.z, 1.0, epsilon = epsilon);
    }

    #[test]
    fn test_morph_point_plane_to_sphere() {
        // Create a reference plane at z=0
        let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

        // Create a target sphere
        let radius = 1.0;
        let target_sphere =
            NurbsSurface::try_sphere(&Point3::origin(), &Vector3::z(), &Vector3::x(), radius)
                .unwrap();

        // Take a point on the reference plane
        let point = Point3::new(0.0, 0.0, 0.0);

        // Morph to the target sphere
        let morphed = point.morph(&ref_surface, &target_sphere, ()).unwrap();

        // The morphed point should be on the sphere
        let distance = morphed.coords.norm();
        assert_relative_eq!(distance, radius, epsilon = 1e-4);
    }
}
