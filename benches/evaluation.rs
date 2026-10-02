//! Evaluation benchmarks: points, derivatives, lengths and tessellation of curves and surfaces.
//!
//! Each case evaluates a batch of parameters spread over the domain, so the numbers reflect the
//! per-call cost of the evaluator (span search, basis functions, allocation) rather than setup.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;
use nalgebra::{Point2, Point3, Point4, Vector2};

const SAMPLES: usize = 1000;

/// A non-rational cubic through a zig-zag of control points, with many knot spans.
fn cubic() -> NurbsCurve3D<f64> {
    let points: Vec<Point3<f64>> = (0..32)
        .map(|i| {
            let x = i as f64;
            Point3::new(x, (x * 0.7).sin() * 3., (x * 0.3).cos())
        })
        .collect();
    NurbsCurve3D::interpolate(&points, 3).unwrap()
}

/// A rational quadratic: the unit circle.
fn circle() -> NurbsCurve2D<f64> {
    NurbsCurve2D::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap()
}

/// A bicubic surface over a 12 × 12 grid of control points.
fn bicubic() -> NurbsSurface3D<f64> {
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

fn parameters(domain: (f64, f64), n: usize) -> Vec<f64> {
    let (a, b) = domain;
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

fn curves(c: &mut Criterion) {
    let cubic = cubic();
    let circle = circle();
    let cubic_ts = parameters(cubic.knots_domain(), SAMPLES);
    let circle_ts = parameters(circle.knots_domain(), SAMPLES);

    c.bench_function("curve/point_at/cubic", |b| {
        b.iter(|| {
            for &t in &cubic_ts {
                black_box(cubic.point_at(black_box(t)));
            }
        })
    });
    c.bench_function("curve/point_at/rational_circle", |b| {
        b.iter(|| {
            for &t in &circle_ts {
                black_box(circle.point_at(black_box(t)));
            }
        })
    });
    for derivs in [1, 2] {
        c.bench_function(&format!("curve/rational_derivatives_{derivs}/cubic"), |b| {
            b.iter(|| {
                for &t in &cubic_ts {
                    black_box(cubic.rational_derivatives(black_box(t), derivs));
                }
            })
        });
    }
    c.bench_function("curve/evaluator/point_at/cubic", |b| {
        b.iter(|| {
            let mut evaluator = cubic.evaluator();
            for &t in &cubic_ts {
                black_box(evaluator.point_at(black_box(t)));
            }
        })
    });
    c.bench_function("curve/evaluator/rational_derivatives_1/cubic", |b| {
        b.iter(|| {
            let mut evaluator = cubic.evaluator();
            for &t in &cubic_ts {
                black_box(evaluator.rational_derivatives(black_box(t), 1));
            }
        })
    });
    c.bench_function("curve/try_length/cubic", |b| {
        b.iter(|| black_box(cubic.try_length().unwrap()))
    });
    c.bench_function("curve/tessellate/rational_circle", |b| {
        b.iter(|| black_box(circle.tessellate(Some(1e-6))))
    });
    let wave = |phase: f64| {
        let points: Vec<Point2<f64>> = (0..32)
            .map(|i| {
                let x = i as f64;
                Point2::new(x, (x * 0.7 + phase).sin() * 3.)
            })
            .collect();
        NurbsCurve2D::interpolate(&points, 3).unwrap()
    };
    let (wave_a, wave_b) = (wave(0.), wave(1.3));
    c.bench_function("curve/find_intersection/cubic", |b| {
        b.iter(|| black_box(wave_a.find_intersection(&wave_b, None).unwrap()))
    });
}

fn surfaces(c: &mut Criterion) {
    let surface = bicubic();
    let side = 32;
    let us = parameters(surface.u_knots_domain(), side);
    let vs = parameters(surface.v_knots_domain(), side);

    c.bench_function("surface/point_at/bicubic", |b| {
        b.iter(|| {
            for &u in &us {
                for &v in &vs {
                    black_box(surface.point_at(black_box(u), black_box(v)));
                }
            }
        })
    });
    c.bench_function("surface/normal_at/bicubic", |b| {
        b.iter(|| {
            for &u in &us {
                for &v in &vs {
                    black_box(surface.normal_at(black_box(u), black_box(v)));
                }
            }
        })
    });
    c.bench_function("surface/evaluator/point_at/bicubic", |b| {
        b.iter(|| {
            let mut evaluator = surface.evaluator();
            for &u in &us {
                for &v in &vs {
                    black_box(evaluator.point_at(black_box(u), black_box(v)));
                }
            }
        })
    });
    c.bench_function("surface/evaluator/normal_at/bicubic", |b| {
        b.iter(|| {
            let mut evaluator = surface.evaluator();
            for &u in &us {
                for &v in &vs {
                    black_box(evaluator.normal_at(black_box(u), black_box(v)));
                }
            }
        })
    });
    c.bench_function("surface/regular_tessellate/bicubic", |b| {
        b.iter(|| black_box(surface.regular_tessellate(side - 1, side - 1)))
    });
}

criterion_group!(benches, curves, surfaces);
criterion_main!(benches);
