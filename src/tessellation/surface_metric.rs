use nalgebra::{Matrix2, Vector2};

use crate::misc::FloatingPoint;

/// First fundamental form (metric tensor) of a parametric surface at a point.
/// Encodes how UV distances map to 3D distances.
///
/// ds² = E·du² + 2F·du·dv + G·dv²
#[derive(Debug, Clone, Copy)]
pub struct SurfaceMetric<T: FloatingPoint> {
    /// E = Su · Su
    pub e: T,
    /// F = Su · Sv
    pub f: T,
    /// G = Sv · Sv
    pub g: T,
}

impl<T: FloatingPoint> SurfaceMetric<T> {
    pub fn new(e: T, f: T, g: T) -> Self {
        Self { e, f, g }
    }

    /// Compute the metric tensor matrix [[E, F], [F, G]]
    pub fn matrix(&self) -> Matrix2<T> {
        Matrix2::new(self.e, self.f, self.f, self.g)
    }

    /// Compute the 3D length of a UV vector under this metric.
    /// |v|² = E·du² + 2F·du·dv + G·dv²
    pub fn length_squared(&self, dv: &Vector2<T>) -> T {
        self.e * dv.x * dv.x
            + T::from_f64(2.0).unwrap() * self.f * dv.x * dv.y
            + self.g * dv.y * dv.y
    }

    /// Compute the 3D length of a UV vector under this metric.
    pub fn length(&self, dv: &Vector2<T>) -> T {
        self.length_squared(dv).sqrt()
    }

    /// Compute the maximum allowable UV step size given a target 3D edge length.
    /// Returns (max_du, max_dv) — the maximum steps in each UV direction
    /// that produce edges of approximately `target_length` in 3D.
    pub fn max_uv_step(&self, target_length: T) -> Vector2<T> {
        let du = if self.e > T::default_epsilon() {
            target_length / self.e.sqrt()
        } else {
            target_length
        };
        let dv = if self.g > T::default_epsilon() {
            target_length / self.g.sqrt()
        } else {
            target_length
        };
        Vector2::new(du, dv)
    }
}

/// Compute the target 3D edge length from curvature and deflection tolerance.
/// Based on chord-height criterion: for a circle of radius R = 1/|k|,
/// the chord length that produces deflection `d` is L = 2·sqrt(2·R·d - d²) ≈ 2·sqrt(2·d/|k|).
pub fn curvature_to_edge_length<T: FloatingPoint>(curvature: T, deflection: T) -> T {
    let eps = T::from_f64(1e-12).unwrap();
    if curvature.abs() < eps {
        return T::from_f64(1e6).unwrap(); // flat → no curvature constraint
    }
    let two = T::from_f64(2.0).unwrap();
    (two * deflection / curvature.abs()).sqrt() * two
}
