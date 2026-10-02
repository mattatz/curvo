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
        let scales = Scales::new(&[size], [parameter]);

        // Check each segment of the curve against the plane
        let tree = CurveBoundingBoxTree::with_divisions(self, options.knot_domain_division);
        let reach = scales.distance(options.minimum_distance);
        let candidates = leaves_reaching_plane(tree, plane, reach)
            .into_iter()
            .filter_map(|node| {
                let curve_segment = node.curve_owned();
                let problem = CurvePlaneIntersectionProblem::new(&curve_segment, plane);
                // from the middle of each leaf
                let (start, end) = curve_segment.knots_domain();
                let init_param = Vector1::new((start + end) * T::from_f64(0.5).unwrap());
                let (param, _) = scales.solve(problem, &options, init_param)?;
                Some(param)
            });

        let distance_at =
            |param: &Vector1<T>| Float::abs(plane.signed_distance(&self.point_at(param[0])));
        let intersections = scales
            .intersections(candidates, 0, options.minimum_distance, distance_at)
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

    /// The bounding box tree is divided at random, so a case is run again and again.
    const RUNS: usize = 20;

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
            for _ in 0..RUNS {
                let intersections = wave.find_intersection(&plane, None).unwrap();
                assert_eq!(intersections.len(), 7, "scaled by {scale}");
            }
        }
    }

    #[test]
    fn a_curve_that_ends_on_a_plane_or_touches_it_intersects_it_once() {
        let line =
            NurbsCurve3D::polyline(&[Point3::new(0., 0., -2.), Point3::new(0., 0., 0.)], false);
        let circle =
            NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
        for _ in 0..RUNS {
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
}
