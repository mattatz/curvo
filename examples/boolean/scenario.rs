use curvo::prelude::{KnotStyle, NurbsCurve2D};
use nalgebra::{Point2, Vector2};

pub fn circle_rectangle_case() -> (NurbsCurve2D<f64>, NurbsCurve2D<f64>) {
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

    (subject, clip)
}

pub fn periodic_interpolation_case() -> (NurbsCurve2D<f64>, NurbsCurve2D<f64>) {
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
    (subject.clone(), subject)
}

pub fn island_case() -> (NurbsCurve2D<f64>, NurbsCurve2D<f64>) {
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
    (subject, clip)
}
