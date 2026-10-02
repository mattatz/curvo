//! Containment benchmarks: whether a curve contains each of a grid of points.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;
use nalgebra::Point2;

mod common;
use common::{circle, wave};

fn contains(c: &mut Criterion) {
    let points: Vec<Point2<f64>> = (0..20)
        .flat_map(|i| {
            (0..20)
                .map(move |j| Point2::new(-1.9 + 3.8 * i as f64 / 19., -1.9 + 3.8 * j as f64 / 19.))
        })
        .collect();
    for (name, curve) in [
        ("rational_circle", circle()),
        ("closed_cubic", closed_wave()),
    ] {
        c.bench_function(&format!("curve/contains/{name}"), |b| {
            b.iter(|| {
                for point in &points {
                    black_box(curve.contains(black_box(point), None).unwrap());
                }
            })
        });
    }
}

/// A closed cubic: a wave along x and back, joined at its ends.
fn closed_wave() -> NurbsCurve2D<f64> {
    let (a, b) = wave(0.).knots_domain();
    let there = wave(0.);
    let points: Vec<Point2<f64>> = (0..=24)
        .map(|i| {
            let t = a + (b - a) * i as f64 / 24.;
            let p = there.point_at(t);
            Point2::new(p.x / 8. - 1.9, p.y / 3. * 0.5)
        })
        .chain((1..24).map(|i| {
            let t = b - (b - a) * i as f64 / 24.;
            let p = there.point_at(t);
            Point2::new(p.x / 8. - 1.9, p.y / 3. * 0.5 + 1.2)
        }))
        .collect();
    NurbsCurve2D::interpolate_periodic(&points, 3, KnotStyle::Centripetal).unwrap()
}

criterion_group!(benches, contains);
criterion_main!(benches);
