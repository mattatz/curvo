use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{
    misc::FloatingPoint,
    prelude::Decompose,
    surface::{NurbsSurface, UVDirection},
};

impl<T: FloatingPoint, D: DimName> Decompose for NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<Vec<NurbsSurface<T, D>>>;

    /// Decompose the surface into a set of Bezier patches of the same degree
    /// Returns a vector of vectors of Bezier patches.
    /// The outer vector is the u direction, the inner vector is the v direction.
    fn try_decompose(&self) -> anyhow::Result<Self::Output> {
        let mut refined = self.clone();

        // Get unique internal knots for u direction
        let u_knot_mults = refined.u_knots().multiplicity();
        let u_internal_knots: Vec<_> = u_knot_mults
            .iter()
            .skip(1) // Skip first knot
            .take(u_knot_mults.len().saturating_sub(2)) // Skip last knot
            .filter(|m| m.multiplicity() < refined.u_degree())
            .map(|m| *m.knot())
            .collect();

        // Insert knots to achieve Bezier multiplicity in u direction
        for knot in u_internal_knots {
            let current_mult = u_knot_mults
                .iter()
                .find(|m| (*m.knot() - knot).abs() < T::default_epsilon())
                .map(|m| m.multiplicity())
                .unwrap_or(0);

            let knots_to_insert = vec![knot; refined.u_degree() - current_mult];
            if !knots_to_insert.is_empty() {
                refined.try_refine_knot(knots_to_insert, UVDirection::U)?;
            }
        }

        // Get unique internal knots for v direction
        let v_knot_mults = refined.v_knots().multiplicity();
        let v_internal_knots: Vec<_> = v_knot_mults
            .iter()
            .skip(1) // Skip first knot
            .take(v_knot_mults.len().saturating_sub(2)) // Skip last knot
            .filter(|m| m.multiplicity() < refined.v_degree())
            .map(|m| *m.knot())
            .collect();

        // Insert knots to achieve Bezier multiplicity in v direction
        for knot in v_internal_knots {
            let current_mult = v_knot_mults
                .iter()
                .find(|m| (*m.knot() - knot).abs() < T::default_epsilon())
                .map(|m| m.multiplicity())
                .unwrap_or(0);

            let knots_to_insert = vec![knot; refined.v_degree() - current_mult];
            if !knots_to_insert.is_empty() {
                refined.try_refine_knot(knots_to_insert, UVDirection::V)?;
            }
        }

        // Get unique knot values after refinement
        let u_unique_knots: Vec<T> = refined
            .u_knots()
            .multiplicity()
            .iter()
            .map(|m| *m.knot())
            .collect();
        let v_unique_knots: Vec<T> = refined
            .v_knots()
            .multiplicity()
            .iter()
            .map(|m| *m.knot())
            .collect();

        // Calculate number of patches
        let u_patches = u_unique_knots.len() - 1;
        let v_patches = v_unique_knots.len() - 1;

        // Extract each Bezier patch
        Ok((0..u_patches)
            .map(|iu| {
                (0..v_patches)
                    .map(|iv| {
                        // Calculate control point indices
                        let u_start = iu * refined.u_degree();
                        let u_end = u_start + refined.u_degree() + 1;
                        let v_start = iv * refined.v_degree();
                        let v_end = v_start + refined.v_degree() + 1;

                        let patch_control_points: Vec<Vec<OPoint<T, D>>> = refined.control_points()
                            [u_start..u_end]
                            .iter()
                            .map(|row| row[v_start..v_end].to_vec())
                            .collect();

                        // Create Bezier knot vectors (normalized to [0, 1])
                        let u_bezier_knots: Vec<T> = (0..2 * (refined.u_degree() + 1))
                            .map(|k| {
                                if k < refined.u_degree() + 1 {
                                    T::zero()
                                } else {
                                    T::one()
                                }
                            })
                            .collect();
                        let v_bezier_knots: Vec<T> = (0..2 * (refined.v_degree() + 1))
                            .map(|k| {
                                if k < refined.v_degree() + 1 {
                                    T::zero()
                                } else {
                                    T::one()
                                }
                            })
                            .collect();

                        NurbsSurface::new(
                            refined.u_degree(),
                            refined.v_degree(),
                            u_bezier_knots,
                            v_bezier_knots,
                            patch_control_points,
                        )
                    })
                    .collect_vec()
            })
            .collect_vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::surface::NurbsSurface3D;

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Point4;

    #[test]
    fn test_bezier_decomposition_simple_surface() {
        // Create a simple bi-cubic surface with 2x2 patches
        let control_points = vec![
            vec![
                Point4::new(0.0, 0.0, 0.0, 1.0),
                Point4::new(1.0, 0.0, 0.0, 1.0),
                Point4::new(2.0, 0.0, 0.0, 1.0),
                Point4::new(3.0, 0.0, 0.0, 1.0),
                Point4::new(4.0, 0.0, 0.0, 1.0),
                Point4::new(5.0, 0.0, 0.0, 1.0),
                Point4::new(6.0, 0.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 1.0, 0.0, 1.0),
                Point4::new(1.0, 1.0, 0.5, 1.0),
                Point4::new(2.0, 1.0, 1.0, 1.0),
                Point4::new(3.0, 1.0, 1.0, 1.0),
                Point4::new(4.0, 1.0, 1.0, 1.0),
                Point4::new(5.0, 1.0, 0.5, 1.0),
                Point4::new(6.0, 1.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 2.0, 0.0, 1.0),
                Point4::new(1.0, 2.0, 0.5, 1.0),
                Point4::new(2.0, 2.0, 1.0, 1.0),
                Point4::new(3.0, 2.0, 1.0, 1.0),
                Point4::new(4.0, 2.0, 1.0, 1.0),
                Point4::new(5.0, 2.0, 0.5, 1.0),
                Point4::new(6.0, 2.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 3.0, 0.0, 1.0),
                Point4::new(1.0, 3.0, 0.5, 1.0),
                Point4::new(2.0, 3.0, 1.0, 1.0),
                Point4::new(3.0, 3.0, 1.0, 1.0),
                Point4::new(4.0, 3.0, 1.0, 1.0),
                Point4::new(5.0, 3.0, 0.5, 1.0),
                Point4::new(6.0, 3.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 4.0, 0.0, 1.0),
                Point4::new(1.0, 4.0, 0.5, 1.0),
                Point4::new(2.0, 4.0, 1.0, 1.0),
                Point4::new(3.0, 4.0, 1.0, 1.0),
                Point4::new(4.0, 4.0, 1.0, 1.0),
                Point4::new(5.0, 4.0, 0.5, 1.0),
                Point4::new(6.0, 4.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 5.0, 0.0, 1.0),
                Point4::new(1.0, 5.0, 0.5, 1.0),
                Point4::new(2.0, 5.0, 1.0, 1.0),
                Point4::new(3.0, 5.0, 1.0, 1.0),
                Point4::new(4.0, 5.0, 1.0, 1.0),
                Point4::new(5.0, 5.0, 0.5, 1.0),
                Point4::new(6.0, 5.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 6.0, 0.0, 1.0),
                Point4::new(1.0, 6.0, 0.0, 1.0),
                Point4::new(2.0, 6.0, 0.0, 1.0),
                Point4::new(3.0, 6.0, 0.0, 1.0),
                Point4::new(4.0, 6.0, 0.0, 1.0),
                Point4::new(5.0, 6.0, 0.0, 1.0),
                Point4::new(6.0, 6.0, 0.0, 1.0),
            ],
        ];

        // Create knot vectors for a bi-cubic surface with internal knots
        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];

        let surface = NurbsSurface3D::<f64>::new(3, 3, u_knots, v_knots, control_points);

        // Decompose into Bezier patches
        let patches = surface.try_decompose().unwrap();
        let patches = patches.iter().flatten().collect_vec();

        // Should get 2x2 = 4 patches
        assert_eq!(patches.len(), 4);

        // Check that each patch is degree 3x3
        for patch in patches.iter() {
            assert_eq!(patch.u_degree(), 3);
            assert_eq!(patch.v_degree(), 3);

            // Each patch should have 4x4 control points
            assert_eq!(patch.control_points().len(), 4);
            assert_eq!(patch.control_points()[0].len(), 4);

            // Check knot vectors are Bezier (all 0s followed by all 1s)
            let u_knots = patch.u_knots().as_slice();
            let v_knots = patch.v_knots().as_slice();

            assert_eq!(u_knots.len(), 8); // 2*(degree+1)
            assert_eq!(v_knots.len(), 8);

            // First 4 knots should be 0, last 4 should be 1
            for i in 0..4 {
                assert_relative_eq!(u_knots[i], 0.0, epsilon = 1e-10);
                assert_relative_eq!(v_knots[i], 0.0, epsilon = 1e-10);
                assert_relative_eq!(u_knots[i + 4], 1.0, epsilon = 1e-10);
                assert_relative_eq!(v_knots[i + 4], 1.0, epsilon = 1e-10);
            }
        }

        // Test that patches evaluate to same points as original surface
        let test_params = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        for u in &test_params {
            for v in &test_params {
                let original_point = surface.point_at(*u, *v);

                // Find which patch contains this parameter
                let patch_u = if *u <= 0.5 { 0 } else { 1 };
                let patch_v = if *v <= 0.5 { 0 } else { 1 };
                let patch_index = patch_u * 2 + patch_v;

                // Map global parameter to local patch parameter
                let local_u = if *u <= 0.5 {
                    *u * 2.0
                } else {
                    (*u - 0.5) * 2.0
                };
                let local_v = if *v <= 0.5 {
                    *v * 2.0
                } else {
                    (*v - 0.5) * 2.0
                };

                let patch_point = patches[patch_index].point_at(local_u, local_v);

                assert_relative_eq!(original_point, patch_point, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_bezier_decomposition_single_patch() {
        // Create a single Bezier patch (no internal knots)
        let control_points = vec![
            vec![
                Point4::new(0.0, 0.0, 0.0, 1.0),
                Point4::new(1.0, 0.0, 0.0, 1.0),
                Point4::new(2.0, 0.0, 0.0, 1.0),
                Point4::new(3.0, 0.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 1.0, 0.0, 1.0),
                Point4::new(1.0, 1.0, 0.5, 1.0),
                Point4::new(2.0, 1.0, 0.5, 1.0),
                Point4::new(3.0, 1.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 2.0, 0.0, 1.0),
                Point4::new(1.0, 2.0, 0.5, 1.0),
                Point4::new(2.0, 2.0, 0.5, 1.0),
                Point4::new(3.0, 2.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 3.0, 0.0, 1.0),
                Point4::new(1.0, 3.0, 0.0, 1.0),
                Point4::new(2.0, 3.0, 0.0, 1.0),
                Point4::new(3.0, 3.0, 0.0, 1.0),
            ],
        ];

        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let surface = NurbsSurface3D::<f64>::new(3, 3, u_knots, v_knots, control_points);

        let patches = surface.try_decompose().unwrap();

        // Should get exactly 1 patch
        assert_eq!(patches.len(), 1);

        // The patch should be identical to the original
        let patch = &patches[0][0];
        assert_eq!(patch.u_degree(), surface.u_degree());
        assert_eq!(patch.v_degree(), surface.v_degree());
        assert_eq!(patch.control_points().len(), surface.control_points().len());
        assert_eq!(
            patch.control_points()[0].len(),
            surface.control_points()[0].len()
        );
    }

    #[test]
    fn test_bezier_decomposition_different_degrees() {
        // Create a surface with different degrees in u and v
        let control_points = vec![
            vec![
                Point4::new(0.0, 0.0, 0.0, 1.0),
                Point4::new(1.0, 0.0, 0.0, 1.0),
                Point4::new(2.0, 0.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 1.0, 0.0, 1.0),
                Point4::new(1.0, 1.0, 0.5, 1.0),
                Point4::new(2.0, 1.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 2.0, 0.0, 1.0),
                Point4::new(1.0, 2.0, 0.5, 1.0),
                Point4::new(2.0, 2.0, 0.0, 1.0),
            ],
            vec![
                Point4::new(0.0, 3.0, 0.0, 1.0),
                Point4::new(1.0, 3.0, 0.0, 1.0),
                Point4::new(2.0, 3.0, 0.0, 1.0),
            ],
        ];

        // Degree 2 in u, degree 1 in v with one internal knot
        let u_knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 1.0, 1.0];

        let surface = NurbsSurface3D::<f64>::new(2, 1, u_knots, v_knots, control_points);

        let patches = surface.try_decompose().unwrap();

        // Should get 2 patches (2 in u direction, 1 in v direction)
        assert_eq!(patches.len(), 2);

        for patch in patches.iter().flatten() {
            assert_eq!(patch.u_degree(), 2);
            assert_eq!(patch.v_degree(), 1);

            // Each patch should have 3x2 control points
            assert_eq!(patch.control_points().len(), 3);
            assert_eq!(patch.control_points()[0].len(), 2);
        }
    }
}
