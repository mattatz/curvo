use approx::assert_relative_eq;
use nalgebra::{Point2, Rotation2, Translation2, Vector2};

use crate::{
    curve::NurbsCurve2D,
    misc::Transformable,
    prelude::PeriodicInterpolation,
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
    let subject = NurbsCurve2D::<f64>::interpolate_periodic(
        &vec![
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
    let subject = NurbsCurve2D::<f64>::interpolate_periodic(
        &vec![
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

/// Degree 3 curve with an unclamped uniform knot vector (issue #104)
fn unclamped_curve() -> crate::curve::NurbsCurve3D<f64> {
    use nalgebra::Point4;
    let square = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)];
    let control_points: Vec<Point4<f64>> = square
        .iter()
        .chain(square[..3].iter())
        .map(|(x, y)| Point4::new(*x, *y, 0.0, 1.0))
        .collect();
    let knots: Vec<f64> = (0..11).map(|i| i as f64 - 7.0).collect();
    crate::curve::NurbsCurve3D::try_new(3, control_points, knots).unwrap()
}

fn sampled_length(curve: &crate::curve::NurbsCurve3D<f64>) -> f64 {
    let (start, end) = curve.knots_domain();
    let n = 10000;
    (0..n)
        .map(|i| {
            let t0 = start + (end - start) * (i as f64) / (n as f64);
            let t1 = start + (end - start) * ((i + 1) as f64) / (n as f64);
            (curve.point_at(t1) - curve.point_at(t0)).norm()
        })
        .sum()
}

#[test]
fn length_of_unclamped_curve() {
    let curve = unclamped_curve();
    assert_eq!(curve.knots_domain(), (-4.0, 0.0));
    assert!(!curve.is_clamped());

    let length = curve.try_length().unwrap();
    assert_relative_eq!(length, sampled_length(&curve), epsilon = 1e-4);
}

#[test]
fn clamp_unclamped_curve() {
    let curve = unclamped_curve();
    let mut clamped = curve.clone();
    clamped.try_clamp().unwrap();

    assert!(clamped.is_clamped());
    assert_eq!(clamped.knots_domain(), curve.knots_domain());

    let (start, end) = curve.knots_domain();
    for i in 0..=100 {
        let t = start + (end - start) * (i as f64) / 100.;
        assert_relative_eq!(curve.point_at(t), clamped.point_at(t), epsilon = 1e-10);
    }
    assert_relative_eq!(
        curve.try_length().unwrap(),
        clamped.try_length().unwrap(),
        epsilon = 1e-8
    );

    // clamping an already clamped curve does nothing
    let mut twice = clamped.clone();
    twice.try_clamp().unwrap();
    assert_eq!(twice.knots().as_slice(), clamped.knots().as_slice());
    assert_eq!(twice.control_points(), clamped.control_points());
}

#[test]
fn an_evaluator_returns_exactly_what_the_curve_does_in_any_parameter_order() {
    use crate::curve::NurbsCurve3D;
    use nalgebra::Point4;
    // Repeated interior knots, so spans of zero length sit between real ones.
    let points: Vec<Point4<f64>> = (0..9)
        .map(|i| {
            let x = i as f64;
            Point4::new(x, x.sin(), x.cos(), 1. + 0.1 * x)
        })
        .collect();
    let knots = vec![0., 0., 0., 0., 1., 2., 2., 2., 3., 4., 4., 4., 4.];
    let curve = NurbsCurve3D::try_new(3, points, knots.clone()).unwrap();
    let (start, end) = curve.knots_domain();

    let mut ts: Vec<f64> = (0..=200)
        .map(|i| start + (end - start) * i as f64 / 200.)
        .collect();
    ts.extend(knots.iter().copied()); // exactly on every knot, including the ends
    ts.extend([start - 1., end + 1., end - 1e-12, start + 1e-12]);
    let mut orders = vec![ts.clone()];
    orders.push(ts.iter().rev().copied().collect());
    // a fixed shuffle
    let mut shuffled = ts.clone();
    for i in 0..shuffled.len() {
        let j = (i * 7919 + 13) % shuffled.len();
        shuffled.swap(i, j);
    }
    orders.push(shuffled);

    for order in orders {
        let mut evaluator = curve.evaluator();
        for &t in &order {
            assert_eq!(evaluator.point_at(t), curve.point_at(t), "t = {t}");
            assert_eq!(
                evaluator.rational_derivatives(t, 2),
                curve.rational_derivatives(t, 2),
                "t = {t}"
            );
        }
    }
}

#[test]
fn knots_out_of_order_or_outside_the_domain_do_not_refine_a_curve() {
    use crate::curve::NurbsCurve3D;
    use nalgebra::Point4;
    let points: Vec<Point4<f64>> = (0..7)
        .map(|i| Point4::new(i as f64, (i as f64).sin(), 0., 1.))
        .collect();
    // unclamped, so there are knots on both sides of the domain
    let knots: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let curve = NurbsCurve3D::try_new(2, points, knots).unwrap();
    let (start, end) = curve.knots_domain();
    let refine = |knots: Vec<f64>| curve.clone().try_refine_knot(knots);

    assert!(refine(vec![]).is_ok());
    assert!(refine(vec![start, 4.5, 4.5, end]).is_ok());
    assert!(refine(vec![4.5, 3.5]).is_err());
    assert!(refine(vec![start - 0.5]).is_err());
    assert!(refine(vec![4.5, end + 0.5]).is_err());
}
