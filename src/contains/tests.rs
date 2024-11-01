use crate::prelude::*;
use nalgebra::{Point2, Rotation2, Translation2, Vector2};

const OPTIONS: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
    minimum_distance: 1e-4,
    knot_domain_division: 500,
    max_iters: 1000,
    step_size_tolerance: 1e-8,
    cost_tolerance: 1e-10,
};

#[test]
fn test_circle_boundary_case() {
    let radius = 1.;
    let circle =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), radius)
            .unwrap();
    assert!(circle
        .contains(&Point2::new(radius, 0.0), Some(OPTIONS))
        .unwrap());
    assert!(circle
        .contains(&Point2::new(0., radius), Some(OPTIONS))
        .unwrap());
    assert!(circle
        .contains(&Point2::new(-radius, 0.), Some(OPTIONS))
        .unwrap());
    assert!(circle
        .contains(&Point2::new(0., -radius), Some(OPTIONS))
        .unwrap());
    assert!(!circle
        .contains(&Point2::new(-radius * 5., radius), Some(OPTIONS))
        .unwrap());
    assert!(!circle
        .contains(&Point2::new(-radius * 5., -radius), Some(OPTIONS))
        .unwrap());
}

#[test]
fn test_rectangle_boundary_case() {
    let dx = 2.;
    let dy = 1.;
    let rectangle = NurbsCurve2D::<f64>::polyline(
        &[
            Point2::new(0., 0.),
            Point2::new(dx, 0.),
            Point2::new(dx, dy),
            Point2::new(0., dy),
            Point2::new(0., 0.),
        ],
        true,
    );
    assert!(rectangle
        .contains(&Point2::new(0., 0.), Some(OPTIONS))
        .unwrap());
    assert!(rectangle
        .contains(&Point2::new(dx, 0.), Some(OPTIONS))
        .unwrap());
    assert!(rectangle
        .contains(&Point2::new(dx, dy), Some(OPTIONS))
        .unwrap());
    assert!(rectangle
        .contains(&Point2::new(0., dy), Some(OPTIONS))
        .unwrap());

    assert!(!rectangle
        .contains(&Point2::new(-dx, 0.), Some(OPTIONS))
        .unwrap());
    assert!(!rectangle
        .contains(&Point2::new(-dx, dy), Some(OPTIONS))
        .unwrap());
}

#[test]
fn test_problem_case() {
    let dx = 2.25;
    let dy = 0.5;

    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &[
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 0.5),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 0.5),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();

    let delta: f64 = 12.13593589026;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = subject.transformed(&trans.into());
    let point = clip.point_at(clip.knots_domain().0);
    let contains = subject.contains(&point, Some(OPTIONS)).unwrap();
    assert!(contains);
}

#[test]
fn test_problem_case_2() {
    let dx = 2.25;
    let dy = 0.5;

    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &[
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 0.5),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 0.5),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();

    let delta: f64 = 5.80492045224;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = subject.transformed(&trans.into());
    let point = clip.point_at(clip.knots_domain().0);

    let contains = subject.contains(&point, Some(OPTIONS)).unwrap();
    assert!(!contains);
}

#[test]
fn test_problem_case_3() {
    let dx = 2.25;
    let dy = 0.5;

    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &[
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 0.5),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 0.5),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();

    let delta: f64 = 3.1613637252600006;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = subject.transformed(&trans.into());
    let point = subject.point_at(subject.knots_domain().0);
    let contains = clip.contains(&point, Some(OPTIONS)).unwrap();
    assert!(!contains);
}
