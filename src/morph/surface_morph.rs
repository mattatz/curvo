use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint};

use crate::{misc::FloatingPoint, morph::Morph, surface::nurbs_surface::NurbsSurface};

// Implementation for NurbsSurface
impl<T> Morph<T, Const<4>> for NurbsSurface<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat,
{
    type Output = NurbsSurface<T, Const<4>>;

    /// Morphs a surface from the reference surface to the target surface.
    fn morph(
        &self,
        reference_surface: &NurbsSurface<T, Const<4>>,
        target_surface: &NurbsSurface<T, Const<4>>,
    ) -> anyhow::Result<Self::Output> {
        // Get dehomogenized control points
        let dehom_control_points = self.dehomogenized_control_points();

        // Get weights from original homogeneous control points
        let weights: Vec<Vec<T>> = self
            .control_points()
            .iter()
            .map(|row| row.iter().map(|pt| pt[3]).collect())
            .collect();

        // Morph each control point
        let morphed_points: anyhow::Result<Vec<Vec<_>>> = dehom_control_points
            .iter()
            .map(|row| {
                row.iter()
                    .map(|pt| pt.morph(reference_surface, target_surface))
                    .collect()
            })
            .collect();
        let morphed_points = morphed_points?;

        // Re-homogenize the control points with original weights
        let homogenized_control_points: Vec<Vec<OPoint<T, Const<4>>>> = morphed_points
            .iter()
            .zip(weights.iter())
            .map(|(row, weight_row)| {
                row.iter()
                    .zip(weight_row.iter())
                    .map(|(pt, &w)| {
                        let mut homogenized = OPoint::<T, Const<4>>::origin();
                        for i in 0..3 {
                            homogenized[i] = pt[i] * w;
                        }
                        homogenized[3] = w;
                        homogenized
                    })
                    .collect()
            })
            .collect();

        // Create a new surface with the morphed control points
        Ok(NurbsSurface::new(
            self.u_degree(),
            self.v_degree(),
            self.u_knots().clone(),
            self.v_knots().clone(),
            homogenized_control_points,
        ))
    }
}
