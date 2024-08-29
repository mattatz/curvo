use argmin::core::ArgminFloat;
use nalgebra::{ComplexField, Const, Point2};

use crate::{
    curve::NurbsCurve,
    misc::{FloatingPoint, Line},
    prelude::{CurveIntersection, CurveIntersectionSolverOptions},
};

/// Find intersections between two curves without degeneracies for clipping algorithm.
pub fn find_intersections_without_degeneracies<T: FloatingPoint + ArgminFloat>(
    a: &NurbsCurve<T, Const<3>>,
    b: &NurbsCurve<T, Const<3>>,
    option: Option<CurveIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<CurveIntersection<Point2<T>, T>>> {
    let intersections = a.find_intersections(b, option)?;
    // println!("# of origin: {}", intersections.len());

    let parameter_eps = T::from_f64(1e-3).unwrap();
    let collinear_dot_threshold = T::one() - T::from_f64(1e-3).unwrap();
    let filtered = intersections
        .into_iter()
        .filter(|it| {
            let a0 = a.point_at(it.a().1 - parameter_eps);
            let a1 = a.point_at(it.a().1 + parameter_eps);
            let la = Line::new(a0, a1);
            let b0 = b.point_at(it.b().1 - parameter_eps);
            let b1 = b.point_at(it.b().1 + parameter_eps);
            let lb = Line::new(b0, b1);
            let intersected = la.intersects(&lb);
            if !intersected {
                false
            } else {
                let dot =
                    ComplexField::abs(la.tangent().normalize().dot(&lb.tangent().normalize()));
                // println!("intersected: {}, dot: {}, {}", intersected, dot, dot < tangent_threshold);
                dot < collinear_dot_threshold
            }
        })
        .collect();

    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use crate::{boolean::degeneracies::find_intersections_without_degeneracies, prelude::*};
    use nalgebra::{Point2, Rotation2, Translation2, Vector2};

    const OPTIONS: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
        minimum_distance: 1e-4,
        knot_domain_division: 500,
        max_iters: 1000,
        step_size_tolerance: 1e-8,
        cost_tolerance: 1e-10,
    };

    #[test]
    fn test_circle_rectangle() {
        let circle =
            NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
                .unwrap();
        let rectangle = NurbsCurve2D::<f64>::polyline(&[Point2::new(0., 2.),
            Point2::new(1., 2.),
            Point2::new(1., -2.),
            Point2::new(0., -2.),
            Point2::new(0., 2.)]);
        let intersections =
            find_intersections_without_degeneracies(&circle, &rectangle, Some(OPTIONS)).unwrap();
        assert_eq!(intersections.len(), 2);

        let rectangle_2 = NurbsCurve2D::<f64>::polyline(&[Point2::new(-0.5, 2.),
            Point2::new(0.5, 2.),
            Point2::new(0.5, -1.),
            Point2::new(-0.5, -1.),
            Point2::new(-0.5, 2.)]);
        let intersections_2 =
            find_intersections_without_degeneracies(&circle, &rectangle_2, Some(OPTIONS)).unwrap();
        assert_eq!(intersections_2.len(), 4);
    }

    #[test]
    fn test_problem_case() {
        let dx = 2.25;
        let dy = 0.5;

        let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
            &[Point2::new(-dx, -dy),
                Point2::new(0., -dy * 0.5),
                Point2::new(dx, -dy),
                Point2::new(dx, dy),
                Point2::new(0., dy * 0.5),
                Point2::new(-dx, dy)],
            3,
            KnotStyle::Centripetal,
        )
        .unwrap();

        let delta: f64 = 2.6065769597400004;
        let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
        let clip = subject.transformed(&trans.into());
        let intersections =
            find_intersections_without_degeneracies(&subject, &clip, Some(OPTIONS)).unwrap();
        assert_eq!(intersections.len(), 4);

        let delta_2: f64 = 3.28621257276;
        let trans_2 = Translation2::new(delta_2.cos(), 0.) * Rotation2::new(delta_2);
        let clip_2 = subject.transformed(&trans_2.into());
        let intersections_2 =
            find_intersections_without_degeneracies(&subject, &clip_2, Some(OPTIONS)).unwrap();
        assert_eq!(intersections_2.len(), 6);
    }
}
