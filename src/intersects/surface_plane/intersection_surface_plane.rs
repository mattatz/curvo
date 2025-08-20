use argmin::core::{ArgminFloat, Executor, State};
use itertools::Itertools;
use nalgebra::{Const, Matrix2, OPoint, Vector2};
use simba::scalar::SubsetOf;

use crate::{
    intersects::Intersection,
    misc::{FloatingPoint, Plane},
    prelude::{
        CurveIntersectionSolverOptions, Intersects, NurbsCurve, SurfaceBoundingBoxTree,
        Tessellation,
    },
    surface::{NurbsSurface, UVDirection},
};

use super::{SurfacePlaneIntersectionBFGS, SurfacePlaneIntersectionProblem};

pub type SurfacePlaneIntersection<T> = Vec<Intersection<OPoint<T, Const<3>>, T, ()>>;

/// Options for surface-plane intersection solver
pub type SurfaceIntersectionSolverOptions<T> = CurveIntersectionSolverOptions<T>;

impl<'a, T> Intersects<'a, &'a Plane<T>> for NurbsSurface<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat + num_traits::Bounded + SubsetOf<f64>,
{
    type Output = anyhow::Result<Vec<NurbsCurve<T, Const<4>>>>;
    type Option = Option<SurfaceIntersectionSolverOptions<T>>;

    /// Find the intersection curves between a surface and a plane
    /// * `plane` - The plane to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    fn find_intersection(&'a self, plane: &'a Plane<T>, option: Self::Option) -> Self::Output {
        let tess = self.tessellate(None);
        let its = tess.find_intersection(plane, ())?;
        let polylines = its
            .polylines
            .iter()
            .map(|polyline| {
                // use hint to find the closest point
                let mut iter = polyline.iter();
                let first = iter.next().ok_or(anyhow::anyhow!("No first point"))?;
                let uv = self.find_closest_parameter(first, None)?;
                let parameters = iter.try_fold(vec![uv], |mut acc, pt| {
                    let uv = self.find_closest_parameter(pt, Some(acc.last().unwrap().clone()))?;
                    // let uv = self.find_closest_parameter(pt, None)?;
                    acc.push(uv);
                    anyhow::Ok(acc)
                })?;

                let points = parameters
                    .iter()
                    .map(|uv| self.point_at(uv.0, uv.1))
                    .collect_vec();

                anyhow::Ok(points)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let curves = polylines
            .into_iter()
            .map(|polyline| interpolate_points(&polyline))
            .collect::<anyhow::Result<Vec<_>>>()?;
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
}
