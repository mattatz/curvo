use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint};

use crate::{
    curve::nurbs_curve::NurbsCurve, misc::FloatingPoint, morph::Morph,
    surface::nurbs_surface::NurbsSurface,
};

// Implementation for NurbsCurve
impl<T> Morph<T, Const<4>> for NurbsCurve<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat,
{
    type Output = NurbsCurve<T, Const<4>>;

    /// Morphs a curve from the reference surface to the target surface.
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, Const<4>>,
        target_surface: &NurbsSurface<T, Const<4>>,
    ) -> anyhow::Result<Self::Output> {
        // Get dehomogenized control points
        let dehom_control_points = self.dehomogenized_control_points();
        let weights = self.weights();

        // Morph each control point
        let morphed_points: anyhow::Result<Vec<_>> = dehom_control_points
            .iter()
            .map(|pt| pt.morph(reference_surface, target_surface))
            .collect();
        let morphed_points = morphed_points?;

        // Re-homogenize the control points with original weights
        let homogenized_control_points: Vec<OPoint<T, Const<4>>> = morphed_points
            .iter()
            .zip(weights.iter())
            .map(|(pt, &w)| {
                let mut homogenized = OPoint::<T, Const<4>>::origin();
                for i in 0..3 {
                    homogenized[i] = pt[i] * w;
                }
                homogenized[3] = w;
                homogenized
            })
            .collect();

        // Create a new curve with the morphed control points
        NurbsCurve::try_new(
            self.degree(),
            homogenized_control_points,
            self.knots().to_vec(),
        )
    }
}
