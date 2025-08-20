use std::f64::consts::{FRAC_PI_2, PI, TAU};

use curvo::prelude::{CompoundCurve, KnotStyle, NurbsCurve2D, PeriodicInterpolation, Region};
use nalgebra::{Point2, Vector2};

use super::CurveVariant;

pub type CurveVariants = (CurveVariant, CurveVariant);

pub fn circle_rectangle_case() -> CurveVariants {
    let subject =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    (subject.into(), rectangle(4.5, 1.))
}

pub fn periodic_interpolation_case() -> CurveVariants {
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
    (subject.clone().into(), subject.into())
}

pub fn island_case() -> CurveVariants {
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
    (subject.into(), rectangle(2., 2.))
}

pub fn compound_circle_x_rectangle_case() -> CurveVariants {
    let rectangle = NurbsCurve2D::polyline(
        &[
            Point2::new(0., 2.),
            Point2::new(0., -2.),
            Point2::new(2., -2.),
            Point2::new(2., 2.),
            Point2::new(0., 2.),
        ],
        false,
    );
    (compound_circle(), rectangle.into())
}

pub fn rounded_rectangle_x_rectangle_case() -> CurveVariants {
    (compound_rounded_rectangle(), rectangle(5., 1.))
}

pub fn rounded_t_shape_x_rectangle_case() -> CurveVariants {
    (compound_rounded_t_shape(), rectangle(5., 0.5))
    // (compound_rounded_t_shape(), rectangle(0.5, 0.5))
}

pub fn rounded_t_shape_x_t_shape_case() -> CurveVariants {
    (compound_rounded_t_shape(), compound_rounded_t_shape())
}

pub fn rectangular_annulus_x_rectangle_case() -> CurveVariants {
    (rectangular_annulus(2.5, 2., 0.75), rectangle(3.2, 0.8))
}

pub fn rectangular_annulus_x_t_shape_case() -> CurveVariants {
    (
        rectangular_annulus(2.5, 2., 0.75),
        compound_rounded_t_shape(),
    )
}

pub fn rectangular_annulus_x_rectangular_annulus_case() -> CurveVariants {
    (
        rectangular_annulus(2.5, 2., 0.75),
        rectangular_annulus(1.5, 1., 0.5),
    )
}

fn rectangle(width: f64, height: f64) -> CurveVariant {
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
        false,
    )
    .into()
}

fn compound_circle() -> CurveVariant {
    let o = Point2::origin();
    let dx = Vector2::x();
    let dy = Vector2::y();
    CompoundCurve::try_new(vec![
        NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
        NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    ])
    .unwrap()
    .into()
}

pub fn compound_rounded_rectangle() -> CurveVariant {
    let length = 2.0;
    let radius = 1.0;
    let dx = Vector2::x();
    let dy = Vector2::y();
    CompoundCurve::try_new(vec![
        NurbsCurve2D::try_arc(
            &Point2::new(length, 0.),
            &dx,
            &dy,
            radius,
            -FRAC_PI_2,
            FRAC_PI_2,
        )
        .unwrap(),
        NurbsCurve2D::polyline(
            &[Point2::new(length, radius), Point2::new(-length, radius)],
            false,
        ),
        NurbsCurve2D::try_arc(
            &Point2::new(-length, 0.),
            &dx,
            &dy,
            radius,
            FRAC_PI_2,
            PI + FRAC_PI_2,
        )
        .unwrap(),
        NurbsCurve2D::polyline(
            &[Point2::new(-length, -radius), Point2::new(length, -radius)],
            false,
        ),
    ])
    .unwrap()
    .into()
}

pub fn compound_rounded_t_shape() -> CurveVariant {
    let w_length = 1.5;
    let h_length = 3.0;
    let radius = 0.5;
    let dx = Vector2::x();
    let dy = Vector2::y();
    CompoundCurve::try_new(vec![
        NurbsCurve2D::try_arc(
            &Point2::new(w_length, 0.),
            &dx,
            &dy,
            radius,
            -FRAC_PI_2,
            FRAC_PI_2,
        )
        .unwrap(),
        NurbsCurve2D::polyline(
            &[
                Point2::new(w_length, radius),
                Point2::new(-w_length, radius),
            ],
            false,
        ),
        NurbsCurve2D::try_arc(
            &Point2::new(-w_length, 0.),
            &dx,
            &dy,
            radius,
            FRAC_PI_2,
            PI + FRAC_PI_2,
        )
        .unwrap(),
        NurbsCurve2D::polyline(
            &[
                Point2::new(-w_length, -radius),
                Point2::new(-radius, -radius),
                Point2::new(-radius, -h_length),
                Point2::new(radius, -h_length),
                Point2::new(radius, -radius),
                Point2::new(w_length, -radius),
            ],
            false,
        ),
    ])
    .unwrap()
    .into()
}

fn rectangular_annulus(width: f64, height: f64, square_size: f64) -> CurveVariant {
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
            false,
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
            false,
        )
        .into()],
    )
    .into()
}
