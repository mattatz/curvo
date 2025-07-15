use std::cmp::Ordering;

use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{Const, Matrix1, OPoint, Vector1};

use crate::{
    bounding_box::BoundingBoxTree,
    curve::NurbsCurve,
    intersects::Intersection,
    misc::{FloatingPoint, Plane},
    prelude::{CurveBoundingBoxTree, CurveIntersectionSolverOptions, HasIntersection, Intersects},
};

use super::{CurvePlaneIntersectionBFGS, CurvePlaneIntersectionProblem};

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

        // Create bounding box tree for the curve
        let tree = CurveBoundingBoxTree::new(
            self,
            Some(
                self.knots_domain_interval() / T::from_usize(options.knot_domain_division).unwrap(),
            ),
        );

        let domain = self.knots_domain();

        // Check each segment of the curve against the plane
        let intersections = collect_leaf_nodes(tree, plane)
            .into_iter()
            .filter_map(|node| {
                let curve_segment = node.curve_owned();
                let problem = CurvePlaneIntersectionProblem::new(&curve_segment, plane);

                // Initial parameter at midpoint of segment
                let segment_domain = curve_segment.knots_domain();
                let init_param = Vector1::<T>::new(
                    (segment_domain.0 + segment_domain.1) * T::from_f64(0.5).unwrap(),
                );

                // Set up solver
                let solver = CurvePlaneIntersectionBFGS::<T>::new()
                    .with_step_size_tolerance(options.step_size_tolerance)
                    .with_cost_tolerance(options.cost_tolerance);

                // Run solver
                let res = Executor::new(problem, solver)
                    .configure(|state| {
                        state
                            .param(init_param)
                            .inv_hessian(Matrix1::identity())
                            .max_iters(options.max_iters)
                    })
                    .run();

                match res {
                    Ok(r) => r.state().get_best_param().and_then(|param| {
                        let t = param[0];
                        if (domain.0..=domain.1).contains(&t) {
                            let point = self.point_at(t);
                            let distance = num_traits::Float::abs(plane.signed_distance(&point));

                            if distance < options.minimum_distance {
                                Some(CurvePlaneIntersection::new((point.clone(), t), (point, ())))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }),
                    Err(_) => None,
                }
            })
            .collect_vec();

        let pts = group_and_extract_closest_intersections(intersections, options.minimum_distance);
        Ok(pts)
    }
}

/// Recursively collect all leaf nodes from a bounding box tree
fn collect_leaf_nodes<'a, 'b, T: FloatingPoint>(
    tree: CurveBoundingBoxTree<'a, T, Const<4>>,
    plane: &'b Plane<T>,
) -> Vec<CurveBoundingBoxTree<'a, T, Const<4>>> {
    let bbox = tree.bounding_box();
    let corners = bbox.corners();

    // Check if bbox straddles the plane
    let distances: Vec<_> = corners.iter().map(|p| plane.signed_distance(p)).collect();
    let has_positive = distances.iter().any(|&d| d > T::zero());
    let has_negative = distances.iter().any(|&d| d < T::zero());

    if !has_positive || !has_negative {
        // Bbox is entirely on one side of the plane
        return vec![];
    }

    if tree.is_dividable() {
        if let Ok((left, right)) = tree.try_divide() {
            let mut nodes = collect_leaf_nodes(left, plane);
            nodes.extend(collect_leaf_nodes(right, plane));
            nodes
        } else {
            vec![tree]
        }
    } else {
        vec![tree]
    }
}

/// Group intersections by parameter and extract unique intersections
fn group_and_extract_closest_intersections<T>(
    intersections: Vec<CurvePlaneIntersection<T>>,
    min_distance: T,
) -> Vec<CurvePlaneIntersection<T>>
where
    T: FloatingPoint + ArgminFloat,
{
    if intersections.is_empty() {
        return vec![];
    }

    let sorted = intersections
        .into_iter()
        .sorted_by(|x, y| x.a().1.partial_cmp(&y.a().1).unwrap_or(Ordering::Equal))
        .collect_vec();

    let groups = sorted
        .into_iter()
        .map(|pt| vec![pt])
        .coalesce(|x, y| {
            let x0 = &x[x.len() - 1];
            let y0 = &y[y.len() - 1];
            let dt = num_traits::Float::abs(x0.a().1 - y0.a().1);
            if dt < min_distance {
                // merge near parameter results
                let group = [x, y].concat();
                Ok(group)
            } else {
                Err((x, y))
            }
        })
        .collect::<Vec<Vec<CurvePlaneIntersection<T>>>>()
        .into_iter()
        .collect_vec();

    groups
        .into_iter()
        .filter_map(|group| match group.len() {
            1 => Some(group[0].clone()),
            _ => {
                // Return the first one (they should all be very close)
                group.into_iter().next()
            }
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use crate::{
        curve::NurbsCurve3D,
        misc::Plane,
        prelude::{HasIntersection, Intersects},
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
        assert_relative_eq!(*pt, Point3::origin());
    }
}
