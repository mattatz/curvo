use std::f64::consts::{PI, TAU};

use nalgebra::{Point2, Vector2, U3};

use crate::{
    curve::NurbsCurve2D,
    prelude::CurveIntersectionSolverOptions,
    region::{CompoundCurve, Region},
};

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
    NurbsCurve2D::polyline(
        &[
            Point2::new(-dx, -dy),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(-dx, dy),
            Point2::new(-dx, -dy),
        ],
        true,
    )
}

fn compound_circle(radius: f64) -> CompoundCurve<f64, U3> {
    let o = Point2::origin();
    let dx = Vector2::x();
    let dy = Vector2::y();
    CompoundCurve::try_new(vec![
        NurbsCurve2D::try_arc(&o, &dx, &dy, radius, 0., PI).unwrap(),
        NurbsCurve2D::try_arc(&o, &dx, &dy, radius, PI, TAU).unwrap(),
    ])
    .unwrap()
}

fn rectangular_annulus(width: f64, height: f64, square_size: f64) -> Region<f64> {
    let dx = width * 0.5;
    let dy = height * 0.5;
    let size = (square_size * 0.5).min(dx).min(dy);
    Region::new(
        NurbsCurve2D::polyline(
            &[
                Point2::new(-dx, -dy),
                Point2::new(dx, -dy),
                Point2::new(dx, dy),
                Point2::new(-dx, dy),
                Point2::new(-dx, -dy),
            ],
            true,
        )
        .into(),
        vec![NurbsCurve2D::polyline(
            &[
                Point2::new(-size, -size),
                Point2::new(size, -size),
                Point2::new(size, size),
                Point2::new(-size, size),
                Point2::new(-size, -size),
            ],
            true,
        )
        .into()],
    )
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
    assert_eq!(region.exterior().spans()[0].curve(), &subject);
    assert_eq!(region.interiors().len(), 0);

    let intersection = subject.intersection(&clip, Some(OPTIONS)).unwrap();
    let regions = intersection.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior().spans()[0].curve(), &clip);
    assert_eq!(region.interiors().len(), 0);

    let diff = subject.difference(&clip, Some(OPTIONS)).unwrap();
    let regions = diff.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior().spans()[0].curve(), &subject);
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(region.interiors()[0].spans()[0].curve(), &clip);
}

/// Test boolean operations between a compound circle (CompoundCurve) and a rectangle
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
    assert_eq!(region.exterior().spans()[0].curve(), &clip);
    assert_eq!(region.interiors().len(), 0);

    let diff = subject.difference(&clip, Some(OPTIONS)).unwrap();
    let regions = diff.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior(), &subject);
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(region.interiors()[0].spans()[0].curve(), &clip);
}

/// Test boolean operations between a rectangular annulus (Region) and a rectangle
#[test]
fn test_rectangular_annulus_x_rectangle() {
    let subject = rectangular_annulus(2., 1.25, 0.25);
    let clip = rectangle(0.5, 0.5);

    let union = subject.union(&clip, Some(OPTIONS)).unwrap();
    let regions = union.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior(), subject.exterior());
    assert_eq!(region.interiors().len(), 0);

    let intersection = subject.intersection(&clip, Some(OPTIONS)).unwrap();
    let regions = intersection.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior().spans()[0].curve(), &clip);
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(region.interiors(), subject.interiors());

    let diff = subject.difference(&clip, Some(OPTIONS)).unwrap();
    let regions = diff.into_regions();
    assert_eq!(regions.len(), 1);
    let region = regions.first().unwrap();
    assert_eq!(region.exterior(), subject.exterior());
    assert_eq!(region.interiors().len(), 1);
    assert_eq!(region.interiors()[0].spans()[0].curve(), &clip);
}

#[test]
#[cfg(feature = "serde")]
fn two_circles_difference_case() {
    use crate::boolean::Boolean;
    let c0 = "{\"spans\":[{\"control_points\":[[0.0,0.0,1.0],[81.31727983645295,0.0,0.7071067811865476],[115.0,115.0,1.0],[81.31727983645297,162.6345596729059,0.7071067811865476],[1.4083438190194563e-14,230.0,1.0],[-81.31727983645295,162.63455967290594,0.7071067811865476],[-115.0,115.00000000000001,1.0],[-81.31727983645298,3.0145775206728485e-14,0.7071067811865476],[-2.8166876380389125e-14,0.0,1.0]],\"degree\":2,\"knots\":[0.0,0.0,0.0,1.5707963267948966,1.5707963267948966,3.141592653589793,3.141592653589793,4.71238898038469,4.71238898038469,6.283185307179586,6.283185307179586,6.283185307179586]}]}";
    let c1 = "{\"spans\":[{\"control_points\":[[0.0,50.0,1.0],[45.961940777125584,35.35533905932738,0.7071067811865476],[65.0,115.0,1.0],[45.96194077712559,127.27922061357856,0.7071067811865476],[7.960204194457796e-15,180.0,1.0],[-45.961940777125584,127.27922061357856,0.7071067811865476],[-65.0,115.00000000000001,1.0],[-45.961940777125605,35.3553390593274,0.7071067811865476],[-1.592040838891559e-14,50.0,1.0]],\"degree\":2,\"knots\":[0.0,0.0,0.0,1.5707963267948966,1.5707963267948966,3.141592653589793,3.141592653589793,4.71238898038469,4.71238898038469,6.283185307179586,6.283185307179586,6.283185307179586]}]}";
    let c0 = serde_json::from_str::<CompoundCurve2D<f64>>(c0).unwrap();
    let c1 = serde_json::from_str::<CompoundCurve2D<f64>>(c1).unwrap();
    assert!(c0.is_closed(None));
    assert!(c1.is_closed(None));
    let diff = c0.difference(&c1, Some(OPTIONS)).unwrap();
    assert_eq!(diff.regions().len(), 1);
    assert!(diff
        .into_regions()
        .first()
        .unwrap()
        .exterior()
        .is_closed(None));
}
