//! Continuity checks for curvatures.

use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OVector};

/// Determine whether two curvatures are G2 continuous.
pub fn is_g2_curvature_continuous<D: DimName>(
    km: OVector<f64, D>,
    kp: OVector<f64, D>,
    cos_angle_tolerance: f64,
    curvature_tolerance: f64,
) -> bool
where
    DefaultAllocator: Allocator<D>,
{
    let relative_tolerance = 0.05;

    let mut cos_kangle_tolerance = cos_angle_tolerance.min(0.999_847_695_156_391_3);
    if cos_kangle_tolerance > 0.95 {
        if cos_angle_tolerance < 0.0 {
            cos_kangle_tolerance = -1.0;
        } else {
            cos_kangle_tolerance = 2.0 * cos_kangle_tolerance * cos_kangle_tolerance - 1.0;
            if cos_angle_tolerance >= 0.0 && cos_kangle_tolerance < 0.0 {
                cos_kangle_tolerance = 0.0;
            }
        }
    }

    !is_curvature_discontinuity(
        km,
        kp,
        cos_kangle_tolerance,
        curvature_tolerance,
        None,
        relative_tolerance,
    )
}

/// Determine if the two curvatures are discontinuous.
fn is_curvature_discontinuity<D: DimName>(
    km: OVector<f64, D>,
    kp: OVector<f64, D>,
    cos_angle_tolerance: f64,
    curvature_tolerance: f64,
    radius_tolerance: Option<f64>,
    relative_tolerance: f64,
) -> bool
where
    DefaultAllocator: Allocator<D>,
{
    let d = (&km - &kp).norm();
    if !d.is_finite() {
        return true;
    }

    if d <= 0.0 || d <= curvature_tolerance {
        return false;
    }

    let zero_curvature = 1e-8;
    let mut km_norm = km.norm();
    let mut kp_norm = kp.norm();

    if !(km_norm > zero_curvature) {
        km_norm = 0.0;
    }
    if !(kp_norm > zero_curvature) {
        kp_norm = 0.0;
        if km_norm == 0.0 {
            return false;
        }
    }

    if !(km_norm > 0.0 && kp_norm > 0.0) {
        return true;
    }

    let mut b_point_of_inflection = curvature_tolerance > 0.0;
    let mut b_different_scalars = b_point_of_inflection;

    if (-1.0..=1.0).contains(&cos_angle_tolerance) {
        let km_o_kp = kp.dot(&km);
        if km_o_kp < km_norm * kp_norm * cos_angle_tolerance {
            return true;
        }
        b_point_of_inflection = false;
    }

    if let Some(radius_tolerance) = radius_tolerance {
        if (km_norm - kp_norm).abs() > kp_norm * km_norm * radius_tolerance {
            return true;
        }
        b_different_scalars = false;
    }

    if relative_tolerance > 0.0 {
        if (km_norm - kp_norm).abs() > km_norm.max(kp_norm) * relative_tolerance {
            return true;
        }
        b_different_scalars = false;
    }

    b_point_of_inflection || b_different_scalars
}
