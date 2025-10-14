pub mod intersection_surface_plane;
pub mod surface_plane_intersection_bfgs;
pub mod surface_plane_intersection_problem;

use argmin::core::{ArgminFloat, Executor, State};
pub use intersection_surface_plane::*;
use nalgebra::{Const, Matrix2, Point3, Vector2};
pub use surface_plane_intersection_bfgs::*;
pub use surface_plane_intersection_problem::*;

use crate::{
    misc::{FloatingPoint, Plane},
    prelude::SurfaceBoundingBoxTree,
    surface::{NurbsSurface3D, UVDirection},
};

/// Find intersection leaf nodes between a surface and a plane
pub fn find_surface_plane_intersection_leaf_nodes<'a, T: FloatingPoint + ArgminFloat>(
    surface: &'a NurbsSurface3D<T>,
    plane: &'a Plane<T>,
    knot_domain_division: usize,
) -> anyhow::Result<Vec<SurfaceBoundingBoxTree<'a, T, Const<4>>>> {
    // Create bounding box tree for the surface
    let tree = SurfaceBoundingBoxTree::new(
        surface,
        UVDirection::U,
        Some({
            let (u_interval, v_interval) = surface.knots_domain_interval();
            let div = T::from_usize(knot_domain_division).unwrap();
            (u_interval / div, v_interval / div)
        }),
    );

    // Check each segment of the surface against the plane
    let leaf_nodes = tree.traverse_leaf_nodes_with_plane(plane);
    Ok(leaf_nodes)
}

/// Find intersection points between a surface and a plane
#[allow(clippy::type_complexity)]
pub fn find_surface_plane_intersection_points<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface3D<T>,
    plane: &Plane<T>,
    options: Option<SurfaceIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<(Point3<T>, (T, T))>> {
    let options = options.unwrap_or_default();

    let leaf_nodes =
        find_surface_plane_intersection_leaf_nodes(surface, plane, options.knot_domain_division)?;

    // Create bounding box tree for the surface
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
                        intersection_points.push((point, (u, v)));
                    }
                }
            }
        }
    }

    Ok(intersection_points)
}
