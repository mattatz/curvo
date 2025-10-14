use super::*;
use crate::prelude::*;
use approx::assert_relative_eq;
use nalgebra::{Point3, Vector3};

#[test]
fn test_morph_curve_plane_to_plane() {
    // Create a reference plane at z=0
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

    // Create a target plane at z=1
    let target_surface =
        NurbsSurface::plane(Point3::new(0.0, 0.0, 1.0), Vector3::x(), Vector3::y());

    // Create a curve on the reference plane
    let points = vec![
        Point3::new(-0.5, -0.5, 0.0),
        Point3::new(0.5, -0.5, 0.0),
        Point3::new(0.5, 0.5, 0.0),
        Point3::new(-0.5, 0.5, 0.0),
    ];
    let curve = NurbsCurve3D::try_interpolate(&points, 3).unwrap();

    // Morph the curve to the target plane
    let morphed_curve = curve.morph(&ref_surface, &target_surface).unwrap();

    // Sample points on both curves
    let (start, end) = morphed_curve.knots_domain();
    let t = (start + end) * 0.5;

    let original_pt = curve.point_at(t);
    let morphed_pt = morphed_curve.point_at(t);

    // The morphed curve should maintain x and y, but have z=1
    assert_relative_eq!(original_pt.x, morphed_pt.x, epsilon = 1e-4);
    assert_relative_eq!(original_pt.y, morphed_pt.y, epsilon = 1e-4);
    assert_relative_eq!(morphed_pt.z, 1.0, epsilon = 1e-4);
}

#[test]
fn test_morph_curve_plane_to_sphere() {
    // Create a reference plane at z=0
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x() * 2.0, Vector3::y() * 2.0);

    // Create a target sphere
    let target_sphere =
        NurbsSurface::try_sphere(&Point3::origin(), &Vector3::z(), &Vector3::x(), 1.0).unwrap();

    // Create a line curve on the reference plane
    let points = vec![Point3::new(-0.5, 0.0, 0.0), Point3::new(0.5, 0.0, 0.0)];
    let curve = NurbsCurve3D::try_interpolate(&points, 1).unwrap();

    // Morph the curve to the sphere
    let morphed_curve = curve.morph(&ref_surface, &target_sphere).unwrap();

    // The morphed curve should have the same parametric structure
    assert_eq!(curve.degree(), morphed_curve.degree());
    assert_eq!(
        curve.control_points().len(),
        morphed_curve.control_points().len()
    );

    // Sample points on the morphed curve should be closer to the sphere than the original
    let (start, end) = morphed_curve.knots_domain();
    let original_avg_distance = (0..10)
        .map(|i| {
            let t = start + (end - start) * (i as f64 / 9.0);
            let pt = curve.point_at(t);
            pt.coords.norm()
        })
        .sum::<f64>()
        / 10.0;

    let morphed_avg_distance = (0..10)
        .map(|i| {
            let t = start + (end - start) * (i as f64 / 9.0);
            let pt = morphed_curve.point_at(t);
            pt.coords.norm()
        })
        .sum::<f64>()
        / 10.0;

    // The morphed curve should be closer to the sphere (radius 1.0)
    assert!(
        (morphed_avg_distance - 1.0).abs() < (original_avg_distance - 1.0).abs(),
        "Morphed curve should be closer to sphere"
    );
}

#[test]
fn test_morph_surface_plane_to_plane() {
    // Create a reference plane at z=0
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x() * 2.0, Vector3::y() * 2.0);

    // Create a target plane at z=1
    let target_surface = NurbsSurface::plane(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::x() * 2.0,
        Vector3::y() * 2.0,
    );

    // Create a smaller plane to morph
    let surface_to_morph = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

    // Morph the surface
    let morphed_surface = surface_to_morph
        .morph(&ref_surface, &target_surface)
        .unwrap();

    // Sample a point on the morphed surface
    let (u_domain, v_domain) = morphed_surface.knots_domain();
    let u = (u_domain.0 + u_domain.1) * 0.5;
    let v = (v_domain.0 + v_domain.1) * 0.5;

    let morphed_pt = morphed_surface.point_at(u, v);

    // The morphed surface should be at z=1
    assert_relative_eq!(morphed_pt.z, 1.0, epsilon = 1e-4);
}

#[test]
fn test_morph_surface_plane_to_sphere() {
    // Create a reference plane at z=0
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x() * 2.0, Vector3::y() * 2.0);

    // Create a target sphere
    let target_sphere =
        NurbsSurface::try_sphere(&Point3::origin(), &Vector3::z(), &Vector3::x(), 1.0).unwrap();

    // Create a smaller plane to morph
    let surface_to_morph =
        NurbsSurface::plane(Point3::origin(), Vector3::x() * 0.5, Vector3::y() * 0.5);

    // Morph the surface
    let morphed_surface = surface_to_morph
        .morph(&ref_surface, &target_sphere)
        .unwrap();

    // The morphed surface should have the same parametric structure
    assert_eq!(surface_to_morph.u_degree(), morphed_surface.u_degree());
    assert_eq!(surface_to_morph.v_degree(), morphed_surface.v_degree());

    // Calculate average distance from origin for both surfaces
    let (u_domain, v_domain) = morphed_surface.knots_domain();
    let mut original_avg_distance = 0.0;
    let mut morphed_avg_distance = 0.0;
    let sample_count = 25;

    for i in 0..5 {
        for j in 0..5 {
            let u = u_domain.0 + (u_domain.1 - u_domain.0) * (i as f64 / 4.0);
            let v = v_domain.0 + (v_domain.1 - v_domain.0) * (j as f64 / 4.0);

            let original_pt = surface_to_morph.point_at(u, v);
            let morphed_pt = morphed_surface.point_at(u, v);

            original_avg_distance += original_pt.coords.norm();
            morphed_avg_distance += morphed_pt.coords.norm();
        }
    }

    original_avg_distance /= sample_count as f64;
    morphed_avg_distance /= sample_count as f64;

    // The morphed surface should be closer to the sphere (radius 1.0)
    assert!(
        (morphed_avg_distance - 1.0).abs() < (original_avg_distance - 1.0).abs(),
        "Morphed surface should be closer to sphere"
    );
}

#[test]
fn test_morph_curve_preserves_degree_and_knots() {
    // Create a reference plane
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

    // Create a target plane
    let target_surface =
        NurbsSurface::plane(Point3::new(0.0, 0.0, 1.0), Vector3::x(), Vector3::y());

    // Create a curve
    let points = vec![
        Point3::new(-0.5, -0.5, 0.0),
        Point3::new(0.5, -0.5, 0.0),
        Point3::new(0.5, 0.5, 0.0),
    ];
    let curve = NurbsCurve3D::try_interpolate(&points, 2).unwrap();

    // Morph the curve
    let morphed_curve = curve.morph(&ref_surface, &target_surface).unwrap();

    // Degree and knot vector should be preserved
    assert_eq!(curve.degree(), morphed_curve.degree());
    assert_eq!(curve.knots().len(), morphed_curve.knots().len());
}

#[test]
fn test_morph_surface_preserves_structure() {
    // Create a reference plane
    let ref_surface = NurbsSurface::plane(Point3::origin(), Vector3::x() * 2.0, Vector3::y() * 2.0);

    // Create a target plane
    let target_surface = NurbsSurface::plane(
        Point3::new(0.0, 0.0, 1.0),
        Vector3::x() * 2.0,
        Vector3::y() * 2.0,
    );

    // Create a surface to morph
    let surface_to_morph = NurbsSurface::plane(Point3::origin(), Vector3::x(), Vector3::y());

    // Morph the surface
    let morphed_surface = surface_to_morph
        .morph(&ref_surface, &target_surface)
        .unwrap();

    // Degrees and knot vectors should be preserved
    assert_eq!(surface_to_morph.u_degree(), morphed_surface.u_degree());
    assert_eq!(surface_to_morph.v_degree(), morphed_surface.v_degree());
    assert_eq!(
        surface_to_morph.u_knots().len(),
        morphed_surface.u_knots().len()
    );
    assert_eq!(
        surface_to_morph.v_knots().len(),
        morphed_surface.v_knots().len()
    );
}
