use approx::assert_relative_eq;
use nalgebra::{Point2, Rotation2, Translation2, Vector2};

use crate::{
    curve::NurbsCurve2D,
    misc::Transformable,
    prelude::{CurveIntersectionSolverOptions, Intersects},
};

use super::KnotStyle;

const OPTIONS: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
    minimum_distance: 1e-4,
    knot_domain_division: 500,
    max_iters: 1000,
    step_size_tolerance: 1e-8,
    cost_tolerance: 1e-10,
};

#[test]
fn problem_case() {
    let dx = 1.25;
    let dy = 1.5;
    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &[
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 1.25),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 1.25),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();
    let clip = NurbsCurve2D::<f64>::polyline(
        &[
            Point2::new(-1., -1.),
            Point2::new(1., -1.),
            Point2::new(1., 1.),
            Point2::new(-1., 1.),
            Point2::new(-1., -1.),
        ],
        true,
    );
    let delta: f64 = 17.58454421724;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = clip.transformed(&trans.into());
    let intersections = subject.find_intersection(&clip, Some(OPTIONS)).unwrap();
    assert_eq!(intersections.len(), 2);
}

#[test]
fn problem_case2() {
    let dx = 1.25;
    let dy = 1.5;
    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &[
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 1.25),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 1.25),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();
    let clip = NurbsCurve2D::<f64>::polyline(
        &[
            Point2::new(-1., -1.),
            Point2::new(1., -1.),
            Point2::new(1., 1.),
            Point2::new(-1., 1.),
            Point2::new(-1., -1.),
        ],
        true,
    );

    let delta: f64 = 1.2782841177;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = clip.transformed(&trans.into());
    let intersections = subject.find_intersection(&clip, Some(OPTIONS)).unwrap();
    assert_eq!(intersections.len(), 2);
}

#[test]
fn greville_abscissae_polyline() {
    let polyline = NurbsCurve2D::<f64>::polyline(
        &[
            Point2::new(0., 0.),
            Point2::new(1., 0.),
            Point2::new(1., 1.),
            Point2::new(0., 1.),
        ],
        false,
    );
    let greville = polyline.greville_abscissae().unwrap();
    polyline
        .control_points()
        .iter()
        .zip(greville.iter())
        .for_each(|(p, t)| {
            let pt = Point2::new(p.x, p.y);
            assert_relative_eq!(polyline.point_at(*t), pt);
        });
}

#[test]
fn greville_abscissae_circle() {
    let curve =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    let greville = curve.greville_abscissae().unwrap();
    curve
        .control_points()
        .iter()
        .zip(greville.iter())
        .for_each(|(p, t)| {
            let pt = Point2::new(p.x, p.y);
            assert_relative_eq!(curve.point_at(*t), pt);
        });
}
