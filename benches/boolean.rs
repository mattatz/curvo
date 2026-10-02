//! Boolean benchmarks: the union, intersection and difference of two curves.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;
use nalgebra::{Point2, Vector2};

mod common;
use common::circle;

fn boolean(c: &mut Criterion) {
    let a = circle();
    let b =
        NurbsCurve2D::try_circle(&Point2::new(1., 0.), &Vector2::x(), &Vector2::y(), 1.).unwrap();
    c.bench_function("curve/union/two_circles", |bench| {
        bench.iter(|| black_box(a.union(&b, None).unwrap()))
    });
    c.bench_function("curve/intersection/two_circles", |bench| {
        bench.iter(|| black_box(a.intersection(&b, None).unwrap()))
    });
    c.bench_function("curve/difference/two_circles", |bench| {
        bench.iter(|| black_box(a.difference(&b, None).unwrap()))
    });
}

criterion_group!(benches, boolean);
criterion_main!(benches);
