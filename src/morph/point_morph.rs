use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint};

use crate::{misc::FloatingPoint, morph::Morph, surface::nurbs_surface::NurbsSurface};

// Implementation for Point3
impl<T> Morph<T, Const<4>> for OPoint<T, Const<3>>
where
    T: FloatingPoint + ArgminFloat,
{
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
    /// let morphed: Point3<f64> = point.morph(&ref_surface, &target_surface).unwrap();
    /// assert_relative_eq!(morphed.z, 1.0, epsilon = 1e-6);
    /// ```
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, Const<4>>,
        target_surface: &NurbsSurface<T, Const<4>>,
    ) -> anyhow::Result<Self::Output> {
        // Find the closest UV parameter on the reference surface
        let (u, v) = reference_surface.find_closest_parameter(self)?;

        // Evaluate the target surface at the same UV parameter
        let morphed_point = target_surface.point_at(u, v);

        Ok(morphed_point)
    }
}
