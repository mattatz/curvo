use std::f64::consts::{PI, TAU};

use curvo::prelude::{CompoundCurve, KnotStyle, NurbsCurve2D};
use nalgebra::{Point2, Vector2};

use super::CurveVariant;

pub type CurveVariants = (CurveVariant, CurveVariant);

pub fn circle_rectangle_case() -> CurveVariants {
    let dx = 2.25;
    let dy = 0.5;
    let subject =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    let clip = NurbsCurve2D::<f64>::polyline(&[
        Point2::new(-dx, -dy),
        Point2::new(dx, -dy),
        Point2::new(dx, dy),
        Point2::new(-dx, dy),
        Point2::new(-dx, -dy),
    ]);

    (subject.into(), clip.into())
}

pub fn periodic_interpolation_case() -> CurveVariants {
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
    (subject.clone().into(), subject.into())
}

pub fn island_case() -> CurveVariants {
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
    let clip = NurbsCurve2D::<f64>::polyline(&[
        Point2::new(-1., -1.),
        Point2::new(1., -1.),
        Point2::new(1., 1.),
        Point2::new(-1., 1.),
        Point2::new(-1., -1.),
    ]);
    (subject.into(), clip.into())
}

pub fn compound_circle_and_rectangle_case() -> CurveVariants {
    let o = Point2::origin();
    let dx = Vector2::x();
    let dy = Vector2::y();
    let compound = CompoundCurve::new(vec![
        NurbsCurve2D::try_arc(&o, &dx, &dy, 1., 0., PI).unwrap(),
        NurbsCurve2D::try_arc(&o, &dx, &dy, 1., PI, TAU).unwrap(),
    ]);
    let rectangle = NurbsCurve2D::polyline(&[
        Point2::new(0., 2.),
        Point2::new(0., -2.),
        Point2::new(2., -2.),
        Point2::new(2., 2.),
        Point2::new(0., 2.),
    ]);
    (compound.into(), rectangle.into())
}
