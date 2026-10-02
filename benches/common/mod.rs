//! The curves and surfaces the benchmarks share.
//!
//! Each benchmark uses only some of them.
#![allow(dead_code)]

use curvo::prelude::*;
use nalgebra::{Point2, Point3, Point4, Vector2};

/// A non-rational cubic through a zig-zag of control points, with many knot spans.
pub fn cubic() -> NurbsCurve3D<f64> {
    let points: Vec<Point3<f64>> = (0..32)
        .map(|i| {
            let x = i as f64;
            Point3::new(x, (x * 0.7).sin() * 3., (x * 0.3).cos())
        })
        .collect();
    NurbsCurve3D::interpolate(&points, 3).unwrap()
}

/// A rational quadratic: the unit circle.
pub fn circle() -> NurbsCurve2D<f64> {
    NurbsCurve2D::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap()
}

/// A bicubic surface over a 12 × 12 grid of control points.
pub fn bicubic() -> NurbsSurface3D<f64> {
    let n = 12;
    let grid = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let (x, y) = (i as f64, j as f64);
                    Point4::new(x, y, (x * 0.5).sin() * (y * 0.4).cos(), 1.)
                })
                .collect()
        })
        .collect();
    let knots = KnotVector::<f64>::uniform(n - 2, 3).to_vec();
    NurbsSurface3D::new(3, 3, knots.clone(), knots, grid)
}

/// A cubic wave over the x axis, crossing the wave of another `phase` several times.
pub fn wave(phase: f64) -> NurbsCurve2D<f64> {
    let points: Vec<Point2<f64>> = (0..32)
        .map(|i| {
            let x = i as f64;
            Point2::new(x, (x * 0.7 + phase).sin() * 3.)
        })
        .collect();
    NurbsCurve2D::interpolate(&points, 3).unwrap()
}

/// A cubic through the bicubic surface again and again.
pub fn piercing() -> NurbsCurve3D<f64> {
    let points: Vec<Point3<f64>> = (0..12)
        .map(|i| {
            let x = i as f64;
            Point3::new(x, 0.5 + x * 0.9, if i % 2 == 0 { 2. } else { -2. })
        })
        .collect();
    NurbsCurve3D::interpolate(&points, 3).unwrap()
}
