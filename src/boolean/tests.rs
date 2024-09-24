use std::f64::consts::{PI, TAU};

use nalgebra::{Point2, Vector2, U3};

use crate::{curve::NurbsCurve2D, prelude::CurveIntersectionSolverOptions, region::CompoundCurve};

use super::Boolean;

const OPTIONS: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
    minimum_distance: 1e-4,
    knot_domain_division: 500,
    max_iters: 1000,
    step_size_tolerance: 1e-8,
    cost_tolerance: 1e-10,
};

fn rectangle(width: f64, height: f64) -> NurbsCurve2D<f64> {
    let dx = width * 0.5;
    let dy = height * 0.5;
    NurbsCurve2D::polyline(&[
        Point2::new(-dx, -dy),
        Point2::new(dx, -dy),
        Point2::new(dx, dy),
        Point2::new(-dx, dy),
        Point2::new(-dx, -dy),
    ])
}

fn compound_circle(radius: f64) -> CompoundCurve<f64, U3> {
    let o = Point2::origin();
    let dx = Vector2::x();
    let dy = Vector2::y();
    CompoundCurve::new(vec![
        NurbsCurve2D::try_arc(&o, &dx, &dy, radius, 0., PI).unwrap(),
        NurbsCurve2D::try_arc(&o, &dx, &dy, radius, PI, TAU).unwrap(),
    ])
}

/// Test boolean operations between a circle and a rectangle
#[test]
fn test_circle_x_rectangle() {
    let subject =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    let clip = rectangle(0.5, 0.5);

    let union = subject.union(&clip, Some(OPTIONS)).unwrap();
    let regions = union.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(&region.exterior().spans()[0], &subject);
    assert_eq!(region.interiors().len(), 0);

    let intersection = subject.intersection(&clip, Some(OPTIONS)).unwrap();
    let regions = intersection.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(&region.exterior().spans()[0], &clip);
    assert_eq!(region.interiors().len(), 0);

    let diff = subject.difference(&clip, Some(OPTIONS)).unwrap();
    let regions = diff.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(&region.exterior().spans()[0], &subject);
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(&region.interiors()[0].spans()[0], &clip);
}

/// Test boolean operations between a compound circle and a rectangle
#[test]
fn test_compound_circle_x_rectangle() {
    let subject = compound_circle(1.);
    let clip = rectangle(0.5, 0.5);

    let union = subject.union(&clip, Some(OPTIONS)).unwrap();
    let regions = union.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior(), &subject);
    assert_eq!(region.interiors().len(), 0);

    let intersection = subject.intersection(&clip, Some(OPTIONS)).unwrap();
    let regions = intersection.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(&region.exterior().spans()[0], &clip);
    assert_eq!(region.interiors().len(), 0);

    let diff = subject.difference(&clip, Some(OPTIONS)).unwrap();
    let regions = diff.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior(), &subject);
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(&region.interiors()[0].spans()[0], &clip);
}
