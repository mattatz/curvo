//! Intersection benchmarks: two curves that cross several times, and a surface with a curve that
//! goes through it again and again.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;

mod common;
use common::{bicubic, piercing, wave};

fn intersections(c: &mut Criterion) {
    let (wave_a, wave_b) = (wave(0.), wave(1.3));
    c.bench_function("curve/find_intersection/cubic", |b| {
        b.iter(|| black_box(wave_a.find_intersection(&wave_b, None).unwrap()))
    });

    let (surface, piercing) = (bicubic(), piercing());
    c.bench_function("surface/find_intersection/bicubic", |b| {
        b.iter(|| black_box(surface.find_intersection(&piercing, None).unwrap()))
    });
}

criterion_group!(benches, intersections);
criterion_main!(benches);
