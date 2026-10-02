use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint, Vector1};
use num_traits::Float;

use crate::{
    bounding_box::leaves_reaching_plane,
    curve::NurbsCurve,
    intersects::{
        solve::{curve_scale, Scales},
        Intersection,
    },
    misc::{FloatingPoint, Plane},
    prelude::{CurveBoundingBoxTree, CurveIntersectionSolverOptions, Intersects},
};

use super::CurvePlaneIntersectionProblem;

pub type CurvePlaneIntersection<T> = Intersection<OPoint<T, Const<3>>, T, ()>;

impl<'a, T> Intersects<'a, &'a Plane<T>> for NurbsCurve<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat,
{
    type Output = anyhow::Result<Vec<CurvePlaneIntersection<T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersection points between a curve and a plane
    /// * `plane` - The plane to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    fn find_intersection(&'a self, plane: &'a Plane<T>, option: Self::Option) -> Self::Output {
        let options = option.unwrap_or_default();

        let (size, parameter) = curve_scale(self);
        let scales = Scales::new(&[size], [parameter], options.minimum_distance);

        // Check each segment of the curve against the plane
        let tree = CurveBoundingBoxTree::with_divisions(self, options.knot_domain_division);
        let candidates = leaves_reaching_plane(tree, plane, scales.minimum_distance())
            .into_iter()
            .flat_map(|node| {
                let curve_segment = node.curve_owned();
                let problem = CurvePlaneIntersectionProblem::new(&curve_segment, plane);
                let found = scales.solve_leaf(&problem, &options, [curve_segment.knots_domain()]);
                found.into_iter().flatten().map(|(param, _)| param)
            });

        let distance_at =
            |param: &Vector1<T>| Float::abs(plane.signed_distance(&self.point_at(param[0])));
        let intersections = scales
            .intersections(candidates, 0, distance_at)
            .into_iter()
            .map(|param| {
                let point = self.point_at(param[0]);
                CurvePlaneIntersection::new((point, param[0]), (point, ()))
            })
            .collect();
        Ok(intersections)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        curve::NurbsCurve3D,
        misc::Plane,
        prelude::{HasIntersection, Interpolation, Intersects},
    };
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_line_plane_intersection() {
        // Create a line from (0, 0, -2) to (0, 0, 2) - crosses XY plane at origin
        let line = NurbsCurve3D::<f64>::try_new(
            1,
            vec![
                Point3::new(0.0, 0.0, -2.0).to_homogeneous().into(),
                Point3::new(0.0, 0.0, 2.0).to_homogeneous().into(),
            ],
            vec![0., 0., 1., 1.],
        )
        .unwrap();

        // Create XY plane (z = 0)
        // Normal is z-axis, constant is 0 for plane passing through origin
        let plane = Plane::new(Vector3::z(), 0.0);

        let intersections = line.find_intersection(&plane, None).unwrap();
        assert_eq!(intersections.len(), 1);

        let pt = &intersections[0].a().0;
        assert_relative_eq!(*pt, Point3::origin(), epsilon = 1e-6);
    }

    #[test]
    fn intersections_do_not_depend_on_the_scale() {
        for scale in [1., 1e-3, 1e3] {
            let points: Vec<Point3<f64>> = (0..32)
                .map(|i| {
                    let x = i as f64;
                    Point3::new(x, (x * 0.7).sin() * 3., (x * 0.3).cos()) * scale
                })
                .collect();
            let wave = NurbsCurve3D::interpolate(&points, 3).unwrap();
            // y = 0.5, scaled
            let plane = Plane::new(Vector3::y(), -0.5 * scale);
            let intersections = wave.find_intersection(&plane, None).unwrap();
            assert_eq!(intersections.len(), 7, "scaled by {scale}");
        }
    }

    /// A wave along x, of the given phase and amplitude in y.
    fn wave(phase: f64, amplitude: f64) -> NurbsCurve3D<f64> {
        let points: Vec<Point3<f64>> = (0..32)
            .map(|i| {
                let x = i as f64;
                Point3::new(x, (x * 0.7 + phase).sin() * amplitude, (x * 0.3).cos())
            })
            .collect();
        NurbsCurve3D::interpolate(&points, 3).unwrap()
    }

    #[test]
    fn every_crossing_of_a_wave_with_a_plane_is_found() {
        // The plane y = c just under the crests of the wave or just over its troughs, where two
        // crossings are next to each other: in one leaf of the tree, or with a leaf starting on
        // the crest between them, where the solver cannot tell which way to go.
        let close = [
            (5.656225963109361, 3.852904357355273, 3.7274072544386616),
            (5.2192839757610345, 2.86639641195568, -2.5915477898548076),
        ];
        // and planes anywhere through waves of any phase
        let anywhere = (0..7).flat_map(|i| {
            [-0.95, -0.5, 0.3, 0.9]
                .into_iter()
                .map(move |height| (i as f64, 3., height * 3.))
        });
        for (phase, amplitude, c) in close.into_iter().chain(anywhere) {
            let wave = wave(phase, amplitude);
            // where the wave changes sides of the plane, along its whole domain
            let (start, end) = wave.knots_domain();
            let above = (0..=8000)
                .map(|i| wave.point_at(start + (end - start) * i as f64 / 8000.).y > c)
                .collect::<Vec<_>>();
            let crossings = above.windows(2).filter(|w| w[0] != w[1]).count();

            let plane = Plane::new(Vector3::y(), -c);
            let intersections = wave.find_intersection(&plane, None).unwrap();
            let at = format!("phase {phase}, amplitude {amplitude}, y = {c}");
            assert_eq!(intersections.len(), crossings, "{at}");
            for intersection in intersections {
                assert!((intersection.a().0.y - c).abs() < 1e-4, "{at}");
            }
        }
    }

    #[test]
    fn a_curve_that_ends_on_a_plane_or_touches_it_intersects_it_once() {
        let line =
            NurbsCurve3D::polyline(&[Point3::new(0., 0., -2.), Point3::new(0., 0., 0.)], false);
        let circle =
            NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
        let ends = line
            .find_intersection(&Plane::new(Vector3::z(), 0.), None)
            .unwrap();
        assert_eq!(ends.len(), 1);
        assert_relative_eq!(ends[0].a().0, Point3::origin(), epsilon = 1e-6);

        // y = 1
        let touches = circle
            .find_intersection(&Plane::new(Vector3::y(), -1.), None)
            .unwrap();
        assert_eq!(touches.len(), 1);
        assert_relative_eq!(touches[0].a().0, Point3::new(0., 1., 0.), epsilon = 1e-2);
    }
}
