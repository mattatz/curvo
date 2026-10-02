//! Evaluation benchmarks: points, derivatives, lengths and tessellation of curves and surfaces.
//!
//! Each case evaluates a batch of parameters spread over the domain, so the numbers reflect the
//! per-call cost of the evaluator (span search, basis functions, allocation) rather than setup.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;

mod common;
use common::{bicubic, circle, cubic};

const SAMPLES: usize = 1000;

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
