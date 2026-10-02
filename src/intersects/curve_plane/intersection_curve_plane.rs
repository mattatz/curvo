use argmin::core::ArgminFloat;
use nalgebra::{Const, OPoint, Vector1};
use num_traits::Float;

use crate::{
    bounding_box::BoundingBoxTree,
    curve::NurbsCurve,
    intersects::{
        solve::{closest_in_groups, curve_scale, Scales},
        Intersection,
    },
    misc::{FloatingPoint, Plane},
    prelude::{CurveBoundingBoxTree, CurveIntersectionSolverOptions, HasIntersection, Intersects},
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
        // relative to the size of the curve
        let minimum_distance = scales.distance(options.minimum_distance);
        let distance = |point: &OPoint<T, Const<3>>| Float::abs(plane.signed_distance(point));

        // Create bounding box tree for the curve
        let tree = CurveBoundingBoxTree::new(
            self,
            Some(
                self.knots_domain_interval() / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );

        // Check each segment of the curve against the plane
        let candidates = collect_leaf_nodes(tree, plane, minimum_distance)
            .into_iter()
            .filter_map(|node| {
                let curve_segment = node.curve_owned();
                let problem = CurvePlaneIntersectionProblem::new(&curve_segment, plane);

                // Initial parameter at midpoint of segment
                let segment_domain = curve_segment.knots_domain();
                let init_param = Vector1::<T>::new(
                    (segment_domain.0 + segment_domain.1) * T::from_f64(0.5).unwrap(),
                );

                // Run solver
                let param = scales.solve(problem, &options, init_param)?;

                // An intersection at the end of the domain is found a hair inside it or a hair
                // outside, so the parameter is clamped rather than refused: how far from the
                // plane the point there is decides.
                let t = self.knots().clamp(self.degree(), param[0]);
                let point = self.point_at(t);
                Some(CurvePlaneIntersection::new((point, t), (point, ())))
            });

        // Two candidates are one intersection when the curve is still on the plane halfway
        // between them. That is so of an intersection found from several leaves, and of the
        // candidates found all along the stretch where the curve touches the plane.
        Ok(closest_in_groups(
            candidates,
            |it| distance(&it.a().0),
            minimum_distance,
            |it| it.a().1,
            scales.is_closed(0),
            |x, y, across| {
                let t = if across {
                    scales.halfway_across(0, x.a().1, y.a().1)
                } else {
                    (x.a().1 + y.a().1) * T::from_f64(0.5).unwrap()
                };
                distance(&self.point_at(t)) < minimum_distance
            },
        ))
    }
}

/// Recursively collect the leaf nodes of a bounding box tree that reach within `tolerance` of the
/// plane, so that a curve ending on the plane or touching it is not left out.
fn collect_leaf_nodes<'a, T: FloatingPoint>(
    tree: CurveBoundingBoxTree<'a, T, Const<4>>,
    plane: &Plane<T>,
    tolerance: T,
) -> Vec<CurveBoundingBoxTree<'a, T, Const<4>>> {
    let bbox = tree.bounding_box();
    let corners = bbox.corners();

    // Check if bbox reaches the plane from both sides
    let distances: Vec<_> = corners.iter().map(|p| plane.signed_distance(p)).collect();
    let above = distances.iter().any(|&d| d >= -tolerance);
    let below = distances.iter().any(|&d| d <= tolerance);

    if !above || !below {
        // Bbox is entirely on one side of the plane
        return vec![];
    }

    if tree.is_dividable() {
        if let Ok((left, right)) = tree.try_divide() {
            let mut nodes = collect_leaf_nodes(left, plane, tolerance);
            nodes.extend(collect_leaf_nodes(right, plane, tolerance));
            nodes
        } else {
            vec![tree]
        }
    } else {
        vec![tree]
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
