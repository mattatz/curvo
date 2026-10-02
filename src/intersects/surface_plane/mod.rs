pub mod intersection_surface_plane;
pub mod surface_plane_intersection_problem;

use argmin::core::ArgminFloat;
pub use intersection_surface_plane::*;
use nalgebra::Point3;
pub use surface_plane_intersection_problem::*;

use crate::{
    bounding_box::leaves_reaching_plane,
    intersects::solve::{surface_scale, Scales},
    misc::{FloatingPoint, Plane},
    prelude::SurfaceBoundingBoxTree,
    surface::{NurbsSurface3D, UVDirection},
};

/// The solver for the intersections between a NURBS surface and a plane.
pub type SurfacePlaneIntersectionBFGS<F> = super::IntersectionBFGS<F>;

/// Find intersection points between a surface and a plane
#[allow(clippy::type_complexity)]
pub fn find_surface_plane_intersection_points<T: FloatingPoint + ArgminFloat>(
    surface: &NurbsSurface3D<T>,
    plane: &Plane<T>,
    options: Option<SurfaceIntersectionSolverOptions<T>>,
) -> anyhow::Result<Vec<(Point3<T>, (T, T))>> {
    let options = options.unwrap_or_default();

    let (size, parameters) = surface_scale(surface);
    let scales = Scales::new(&[size], parameters, options.minimum_distance);
    let minimum_distance = scales.minimum_distance();

    // Check each leaf of the surface that reaches the plane
    let tree = SurfaceBoundingBoxTree::with_divisions(
        surface,
        UVDirection::U,
        options.knot_domain_division,
    );
    let intersection_points = leaves_reaching_plane(tree, plane, minimum_distance)
        .into_iter()
        .flat_map(|node| {
            let surface_segment = node.surface_owned();
            let problem = SurfacePlaneIntersectionProblem::new(&surface_segment, plane);
            let (u, v) = surface_segment.knots_domain();
            scales.solve_leaf(&problem, &options, [u, v])
        })
        .flatten()
        .filter_map(|(param, _)| {
            let point = surface.point_at(param[0], param[1]);
            let distance = num_traits::Float::abs(plane.signed_distance(&point));
            (distance < minimum_distance).then_some((point, (param[0], param[1])))
        })
        .collect();

    Ok(intersection_points)
}
