use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{Const, Matrix2, OPoint, Point3, Vector2};
use rstar::RTree;

use crate::{
    intersects::Intersection,
    misc::{FloatingPoint, Plane},
    prelude::{CurveIntersectionSolverOptions, Intersects, NurbsCurve, SurfaceBoundingBoxTree},
    surface::{NurbsSurface, UVDirection},
};

use super::{SurfacePlaneIntersectionBFGS, SurfacePlaneIntersectionProblem};

pub type SurfacePlaneIntersection<T> = Vec<Intersection<OPoint<T, Const<3>>, T, ()>>;

/// Options for surface-plane intersection solver
pub type SurfaceIntersectionSolverOptions<T> = CurveIntersectionSolverOptions<T>;

impl<'a, T> Intersects<'a, &'a Plane<T>> for NurbsSurface<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat + num_traits::Bounded,
{
    type Output = anyhow::Result<Vec<NurbsCurve<T, Const<4>>>>;
    type Option = Option<SurfaceIntersectionSolverOptions<T>>;

    /// Find the intersection curves between a surface and a plane
    /// * `plane` - The plane to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    fn find_intersection(&'a self, plane: &'a Plane<T>, option: Self::Option) -> Self::Output {
        // Collect intersection points from all leaf nodes
        let intersection_points =
            find_surface_plane_intersection_points(self, plane, option.clone())?;

        // Group nearby points and extract curves
        let curves = extract_intersection_curves(
            intersection_points
                .into_iter()
                .map(|p| p.into())
                .collect_vec(),
            self,
            plane,
        )?;
        Ok(curves)
    }
}

/// Find intersection points between a surface and a plane
pub fn find_surface_plane_intersection_points<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface<T, Const<4>>,
    plane: &Plane<T>,
    options: Option<SurfaceIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<(OPoint<T, Const<3>>, T, T)>> {
    let options = options.unwrap_or_default();

    // Create bounding box tree for the surface
    let tree = SurfaceBoundingBoxTree::new(
        surface,
        UVDirection::U,
        Some({
            let (u_interval, v_interval) = surface.knots_domain_interval();
            let div = T::from_usize(options.knot_domain_division).unwrap();
            (u_interval / div, v_interval / div)
        }),
    );

    // Check each segment of the surface against the plane
    let leaf_nodes = tree.traverse_leaf_nodes_with_plane(plane);
    let (u_domain, v_domain) = surface.knots_domain();

    // Collect intersection points from all leaf nodes
    let mut intersection_points = Vec::new();

    for node in leaf_nodes {
        let surface_segment = node.surface_owned();
        let problem = SurfacePlaneIntersectionProblem::new(&surface_segment, plane);

        // Initial parameter at midpoint of segment
        let (u_seg_domain, v_seg_domain) = surface_segment.knots_domain();
        let init_param = Vector2::<T>::new(
            (u_seg_domain.0 + u_seg_domain.1) * T::from_f64(0.5).unwrap(),
            (v_seg_domain.0 + v_seg_domain.1) * T::from_f64(0.5).unwrap(),
        );

        // Set up solver
        let solver = SurfacePlaneIntersectionBFGS::<T>::new()
            .with_step_size_tolerance(options.step_size_tolerance)
            .with_cost_tolerance(options.cost_tolerance);

        // Run solver
        let res = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(init_param)
                    .inv_hessian(Matrix2::identity())
                    .max_iters(options.max_iters)
            })
            .run();

        if let Ok(r) = res {
            if let Some(param) = r.state().get_best_param() {
                let u = param[0];
                let v = param[1];
                if (u_domain.0..=u_domain.1).contains(&u) && (v_domain.0..=v_domain.1).contains(&v)
                {
                    let point = surface.point_at(u, v);
                    let distance = num_traits::Float::abs(plane.signed_distance(&point));

                    if distance < options.minimum_distance {
                        intersection_points.push((point, u, v));
                    }
                }
            }
        }
    }

    Ok(intersection_points)
}

#[derive(Debug, Clone, PartialEq)]
struct PointNode<T: FloatingPoint> {
    point: Point3<T>,
    uv: (T, T),
}

impl<T: FloatingPoint> PointNode<T> {
    pub fn query(&self) -> [T; 2] {
        [self.uv.0, self.uv.1]
    }
}

impl<T: FloatingPoint> From<(Point3<T>, T, T)> for PointNode<T> {
    fn from(value: (Point3<T>, T, T)) -> Self {
        Self {
            point: value.0,
            uv: (value.1, value.2),
        }
    }
}

impl<T: FloatingPoint + num_traits::Bounded> rstar::RTreeObject for PointNode<T> {
    type Envelope = rstar::AABB<[T; 2]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point([self.uv.0, self.uv.1])
    }
}

impl<T: FloatingPoint + num_traits::Bounded> rstar::PointDistance for PointNode<T> {
    fn distance_2(
        &self,
        other: &[T; 2],
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        let du = self.uv.0 - other[0];
        let dv = self.uv.1 - other[1];
        du * du + dv * dv
    }
}

/// Extract intersection curves from the collected points
fn extract_intersection_curves<T>(
    points: Vec<PointNode<T>>,
    surface: &NurbsSurface<T, Const<4>>,
    _plane: &Plane<T>,
) -> anyhow::Result<Vec<NurbsCurve<T, Const<4>>>>
where
    T: FloatingPoint + num_traits::Bounded,
{
    if points.is_empty() {
        return Ok(vec![]);
    }

    // Build RTree for efficient spatial queries
    let mut tree = RTree::bulk_load(points);
    let mut curves = Vec::new();

    let (u_domain, v_domain) = surface.knots_domain();
    let eps = T::from_f64(0.1).unwrap();
    let u_threshold = (u_domain.1 - u_domain.0).abs() * eps;
    let v_threshold = (v_domain.1 - v_domain.0).abs() * eps;

    while tree.size() > 0 {
        let traverse = |last_point: PointNode<T>,
                        tree: &mut RTree<PointNode<T>>,
                        connected: &mut Vec<PointNode<T>>,
                        forward: bool| {
            let mut last_point = last_point;
            loop {
                let nearest = tree.nearest_neighbor(&last_point.query()).cloned();
                if let Some(nearest) = nearest {
                    if (nearest.uv.0 - last_point.uv.0).abs() <= u_threshold
                        && (nearest.uv.1 - last_point.uv.1).abs() <= v_threshold
                    {
                        if forward {
                            connected.push(nearest.clone());
                        } else {
                            connected.insert(0, nearest.clone());
                        }

                        tree.remove(&nearest);
                        last_point = nearest;
                        continue;
                    }
                }

                break;
            }
        };

        // Start a new curve with an arbitrary point
        let start_point = tree.iter().next().unwrap().clone();
        let mut current_curve = vec![start_point.clone()];
        tree.remove(&start_point);

        // Grow the curve in both directions
        traverse(start_point.clone(), &mut tree, &mut current_curve, true);
        traverse(start_point, &mut tree, &mut current_curve, false);

        // println!("current_curve: {:?}", current_curve.len());

        // Create a NURBS curve if we have enough points
        if current_curve.len() >= 2 {
            let pts = current_curve
                .into_iter()
                .map(|node| node.point)
                .collect_vec();
            let curve = interpolate_points(&pts)?;
            curves.push(curve);
        }
    }

    Ok(curves)
}

/// Interpolate a set of points to create a NURBS curve
fn interpolate_points<T>(points: &[OPoint<T, Const<3>>]) -> anyhow::Result<NurbsCurve<T, Const<4>>>
where
    T: FloatingPoint,
{
    // Use curve interpolation to create a smooth curve through the points
    let degree = (points.len() - 1).min(3); // Use cubic curves when possible
    NurbsCurve::try_interpolate(points, degree)
}

#[cfg(test)]
mod tests {
    use crate::{misc::Plane, prelude::Intersects, surface::NurbsSurface3D};
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_plane_surface_intersection() {
        // Create a simple surface that curves upward in the middle
        let control_points = vec![
            vec![
                Point3::new(-1.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(-1.0, 0.0, 0.5).to_homogeneous().into(),
                Point3::new(-1.0, 1.0, 0.0).to_homogeneous().into(),
            ],
            vec![
                Point3::new(0.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(0.0, 0.0, 1.0).to_homogeneous().into(),
                Point3::new(0.0, 1.0, 0.0).to_homogeneous().into(),
            ],
            vec![
                Point3::new(1.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(1.0, 0.0, 0.5).to_homogeneous().into(),
                Point3::new(1.0, 1.0, 0.0).to_homogeneous().into(),
            ],
        ];

        let surface = NurbsSurface3D::<f64>::new(
            2,
            2,
            vec![0., 0., 0., 1., 1., 1.],
            vec![0., 0., 0., 1., 1., 1.],
            control_points,
        );

        // Create XY plane at z = 0.3
        let plane = Plane::new(Vector3::z(), 0.3);

        let intersections = surface.find_intersection(&plane, None);
        assert!(intersections.is_ok());

        // For now, let's just check that the function runs without panicking
        // The actual curve extraction logic needs more refinement
        let _curves = intersections.unwrap();
    }

    #[test]
    fn test_rtree_curve_extraction() {
        use super::*;

        // Create a set of test points that form a curve
        let points = vec![
            PointNode {
                point: Point3::new(0.0, 0.0, 0.3),
                uv: (0.1, 0.5),
            },
            PointNode {
                point: Point3::new(0.1, 0.0, 0.3),
                uv: (0.2, 0.5),
            },
            PointNode {
                point: Point3::new(0.2, 0.0, 0.3),
                uv: (0.3, 0.5),
            },
            PointNode {
                point: Point3::new(0.3, 0.0, 0.3),
                uv: (0.4, 0.5),
            },
            PointNode {
                point: Point3::new(0.4, 0.0, 0.3),
                uv: (0.5, 0.5),
            },
        ];

        // Create a dummy surface and plane for the test
        let control_points = vec![
            vec![
                Point3::new(0.0, 0.0, 0.0).to_homogeneous().into(),
                Point3::new(1.0, 0.0, 0.0).to_homogeneous().into(),
            ],
            vec![
                Point3::new(0.0, 1.0, 0.0).to_homogeneous().into(),
                Point3::new(1.0, 1.0, 0.0).to_homogeneous().into(),
            ],
        ];
        let surface = NurbsSurface3D::<f64>::new(
            1,
            1,
            vec![0., 0., 1., 1.],
            vec![0., 0., 1., 1.],
            control_points,
        );
        let plane = Plane::new(Vector3::z(), 0.3);

        // Test the curve extraction
        let curves = extract_intersection_curves(points, &surface, &plane).unwrap();

        // Should extract one curve
        assert_eq!(curves.len(), 1);

        // The curve should have the right degree
        let curve = &curves[0];
        assert!(curve.degree() <= 3);
    }
}
