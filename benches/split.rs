//! Split benchmarks: a curve at a parameter, a surface along u and along v.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use curvo::prelude::*;

mod common;
use common::{bicubic, cubic};

fn splits(c: &mut Criterion) {
    let cubic = cubic();
    let (start, end) = cubic.knots_domain();
    c.bench_function("curve/try_split/cubic", |b| {
        b.iter(|| {
            black_box(
                cubic
                    .try_split(black_box(start + (end - start) * 0.37))
                    .unwrap(),
            )
        })
    });

    let surface = bicubic();
    for (name, direction) in [("u", UVDirection::U), ("v", UVDirection::V)] {
        c.bench_function(&format!("surface/try_split_{name}/bicubic"), |b| {
            b.iter(|| {
                let option = SplitSurfaceOption::new(black_box(4.3), direction);
                black_box(surface.try_split(option).unwrap())
            })
        });
    }
}

criterion_group!(benches, splits);
criterion_main!(benches);
