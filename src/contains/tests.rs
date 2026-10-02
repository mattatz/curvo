use std::f64::consts::{PI, TAU};

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

    let subject = NurbsCurve2D::<f64>::interpolate_periodic(
        &vec![
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

    let subject = NurbsCurve2D::<f64>::interpolate_periodic(
        &vec![
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

    let subject = NurbsCurve2D::<f64>::interpolate_periodic(
        &vec![
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

#[test]
fn a_point_is_contained_alike_at_any_scale() {
    for scale in [1., 1e-3, 1e3] {
        let scaling = nalgebra::Matrix3::new_scaling(scale);
        let circle =
            NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
                .unwrap()
                .transformed(&scaling);
        assert!(circle.is_closed(), "scaled by {scale}");
        let halves = CompoundCurve::try_new(vec![
            NurbsCurve2D::try_arc(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1., 0., PI)
                .unwrap()
                .transformed(&scaling),
            NurbsCurve2D::try_arc(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1., PI, TAU)
                .unwrap()
                .transformed(&scaling),
        ])
        .unwrap();

        // all around the circle: far inside and outside, and 0.0002 of the radius from it
        for i in 0..90 {
            let angle = TAU * i as f64 / 90.;
            let direction = Vector2::new(angle.cos(), angle.sin());
            for (offset, inside) in [(-0.3, true), (0.3, false), (-2e-4, true), (2e-4, false)] {
                let point = Point2::from(direction * (1. + offset) * scale);
                let at = format!("scaled by {scale}, at {angle} and {offset} of the radius");
                assert_eq!(circle.contains(&point, None).unwrap(), inside, "{at}");
                assert_eq!(halves.contains(&point, None).unwrap(), inside, "{at}");
            }
        }
    }
}

#[test]
fn a_point_next_to_the_top_or_the_bottom_of_a_curve_is_contained_or_not() {
    // A ray from there along x only just crosses the circle, within one leaf of its tree.
    let circle =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    for i in 0..200 {
        for vertical in [0.5 * PI, -0.5 * PI] {
            let angle = vertical + (i as f64 / 200. - 0.5) * 0.1;
            let direction = Vector2::new(angle.cos(), angle.sin());
            for (offset, inside) in [(-2e-4, true), (2e-4, false), (-2e-3, true), (2e-3, false)] {
                let point = Point2::from(direction * (1. + offset));
                let at = format!("at {angle} and {offset} of the radius");
                assert_eq!(circle.contains(&point, None).unwrap(), inside, "{at}");
            }
        }
    }
}
