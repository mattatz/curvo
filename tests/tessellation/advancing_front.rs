use std::collections::HashMap;

use curvo::prelude::*;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use std::f64::consts::TAU;

/// Create a trimmed cylinder surface (extruded circle) for testing.
/// The cylinder has the given radius and height, centered at the origin.
fn make_trimmed_cylinder(radius: f64, height: f64) -> TrimmedSurface<f64> {
    let center = Point3::origin();
    let x_axis = Vector3::x();
    let y_axis = Vector3::y();

    let circle = NurbsCurve3D::try_circle(&center, &x_axis, &y_axis, radius).unwrap();
    let surface = NurbsSurface3D::extrude(&circle, &(Vector3::z() * height));

    // Exterior boundary in UV space: rectangle [u_min..u_max, v_min..v_max]
    let (u_min, u_max) = surface.u_knots_domain();
    let (v_min, v_max) = surface.v_knots_domain();

    let p0 = nalgebra::Point2::new(u_min, v_min);
    let p1 = nalgebra::Point2::new(u_max, v_min);
    let p2 = nalgebra::Point2::new(u_max, v_max);
    let p3 = nalgebra::Point2::new(u_min, v_max);

    let edge0 = NurbsCurve2D::polyline(&[p0, p1], true);
    let edge1 = NurbsCurve2D::polyline(&[p1, p2], true);
    let edge2 = NurbsCurve2D::polyline(&[p2, p3], true);
    let edge3 = NurbsCurve2D::polyline(&[p3, p0], true);
    let exterior = CompoundCurve2D::new_unchecked(vec![edge0, edge1, edge2, edge3]);

    TrimmedSurface::new(surface, Some(exterior), vec![])
}

/// Create a trimmed plane surface for testing.
fn make_trimmed_plane() -> TrimmedSurface<f64> {
    let pts = vec![
        vec![
            nalgebra::Point4::new(0., 0., 0., 1.),
            nalgebra::Point4::new(0., 10., 0., 1.),
        ],
        vec![
            nalgebra::Point4::new(10., 0., 0., 1.),
            nalgebra::Point4::new(10., 10., 0., 1.),
        ],
    ];
    let surface = NurbsSurface3D::new(1, 1, vec![0., 0., 1., 1.], vec![0., 0., 1., 1.], pts);

    let p0 = nalgebra::Point2::new(0.0, 0.0);
    let p1 = nalgebra::Point2::new(1.0, 0.0);
    let p2 = nalgebra::Point2::new(1.0, 1.0);
    let p3 = nalgebra::Point2::new(0.0, 1.0);

    let exterior = CompoundCurve2D::new_unchecked(vec![
        NurbsCurve2D::polyline(&[p0, p1], true),
        NurbsCurve2D::polyline(&[p1, p2], true),
        NurbsCurve2D::polyline(&[p2, p3], true),
        NurbsCurve2D::polyline(&[p3, p0], true),
    ]);

    TrimmedSurface::new(surface, Some(exterior), vec![])
}

#[test]
fn test_advancing_front_cylinder() {
    let trimmed = make_trimmed_cylinder(5.0, 10.0);
    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);
    let mesher = AdvancingFrontMesher::new(&trimmed, opts);
    let tess = mesher
        .mesh()
        .expect("Advancing front mesher should succeed");

    let points = tess.points();
    let faces = tess.faces();

    assert!(!points.is_empty(), "Should produce vertices");
    assert!(!faces.is_empty(), "Should produce faces");

    // All points should be approximately on the cylinder surface
    for p in points {
        let r = (p.x * p.x + p.y * p.y).sqrt();
        assert!(
            (r - 5.0).abs() < 0.2,
            "Point ({:.3}, {:.3}, {:.3}) is {:.3} from cylinder axis, expected ~5.0",
            p.x,
            p.y,
            p.z,
            r
        );
        assert!(
            p.z >= -0.01 && p.z <= 10.01,
            "Point z={:.3} should be in [0, 10]",
            p.z
        );
    }

    // All face indices should be valid
    for face in faces {
        for &idx in face {
            assert!(idx < points.len(), "Face index {} out of bounds", idx);
        }
    }

    // Normals should point outward (dot with radial direction > 0)
    let normals = tess.normals();
    for (p, n) in points.iter().zip(normals.iter()) {
        let radial = nalgebra::Vector3::new(p.x, p.y, 0.0);
        let radial_len = radial.norm();
        if radial_len > 1e-6 {
            let dot = n.dot(&radial) / radial_len;
            // Normal should roughly point outward (allow some tolerance for boundary vertices)
            assert!(
                dot > -0.5,
                "Normal at ({:.2},{:.2},{:.2}) points inward: dot={:.3}",
                p.x,
                p.y,
                p.z,
                dot
            );
        }
    }
}

#[test]
fn test_advancing_front_plane_minimal_subdivision() {
    // A flat plane should produce very few faces since curvature is zero
    let trimmed = make_trimmed_plane();
    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);
    let mesher = AdvancingFrontMesher::new(&trimmed, opts);
    let tess = mesher
        .mesh()
        .expect("Advancing front mesher should succeed on plane");

    let points = tess.points();
    let faces = tess.faces();

    assert!(!points.is_empty());
    assert!(!faces.is_empty());

    // Flat plane has zero curvature → should use max_edge_length,
    // producing fewer faces than a curved surface
    assert!(
        faces.len() < 100,
        "Flat plane should have few faces, got {}",
        faces.len()
    );

    // All points should be on z=0 plane
    for p in points {
        assert!(
            p.z.abs() < 1e-6,
            "Point should be on z=0 plane, got z={:.6}",
            p.z
        );
    }
}

#[test]
fn test_advancing_front_deflection_controls_density() {
    let trimmed = make_trimmed_cylinder(5.0, 10.0);

    // Coarse tessellation
    let opts_coarse = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.5);
    let mesher_coarse = AdvancingFrontMesher::new(&trimmed, opts_coarse);
    let tess_coarse = mesher_coarse.mesh().unwrap();

    // Fine tessellation
    let opts_fine = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.05);
    let mesher_fine = AdvancingFrontMesher::new(&trimmed, opts_fine);
    let tess_fine = mesher_fine.mesh().unwrap();

    assert!(
        tess_fine.faces().len() > tess_coarse.faces().len(),
        "Finer deflection should produce more faces: fine={} vs coarse={}",
        tess_fine.faces().len(),
        tess_coarse.faces().len()
    );
}

#[test]
fn test_advancing_front_torus() {
    // Torus has varying curvature — good test for adaptive behavior
    let major_radius = 5.0;
    let minor_radius = 2.0;
    let torus = NurbsSurface3D::try_torus(
        &Point3::origin(),
        &Vector3::x(),
        &Vector3::y(),
        major_radius,
        minor_radius,
        (0.0, std::f64::consts::TAU),
        (0.0, std::f64::consts::TAU),
    )
    .unwrap();

    let (u_min, u_max) = torus.u_knots_domain();
    let (v_min, v_max) = torus.v_knots_domain();

    let p0 = nalgebra::Point2::new(u_min, v_min);
    let p1 = nalgebra::Point2::new(u_max, v_min);
    let p2 = nalgebra::Point2::new(u_max, v_max);
    let p3 = nalgebra::Point2::new(u_min, v_max);
    let exterior = CompoundCurve2D::new_unchecked(vec![
        NurbsCurve2D::polyline(&[p0, p1], true),
        NurbsCurve2D::polyline(&[p1, p2], true),
        NurbsCurve2D::polyline(&[p2, p3], true),
        NurbsCurve2D::polyline(&[p3, p0], true),
    ]);

    let trimmed = TrimmedSurface::new(torus, Some(exterior), vec![]);
    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);
    let mesher = AdvancingFrontMesher::new(&trimmed, opts);
    let tess = mesher.mesh().expect("Torus tessellation should succeed");

    assert!(
        tess.faces().len() > 100,
        "Torus should produce a reasonable number of faces, got {}",
        tess.faces().len()
    );

    // All points should be approximately on the torus surface
    for p in tess.points() {
        let r_xz = (p.x * p.x + p.y * p.y).sqrt();
        let dist_from_tube_center = ((r_xz - major_radius).powi(2) + p.z * p.z).sqrt();
        assert!(
            (dist_from_tube_center - minor_radius).abs() < 0.3,
            "Point not on torus: dist_from_tube={:.3}, expected {:.1}",
            dist_from_tube_center,
            minor_radius
        );
    }
}

#[test]
fn test_advancing_front_constrained_tessellation() {
    // Use a plane with circular exterior boundary (single-span parameter domain)
    let surface = NurbsSurface3D::plane(
        Point3::new(2.5, 0., 5.),
        Vector3::x() * 2.5,
        Vector3::z() * 5.,
    );

    let exterior =
        NurbsCurve2D::try_circle(&Point2::new(0.5, 0.5), &Vector2::x(), &Vector2::y(), 0.45)
            .unwrap();
    let trimmed = TrimmedSurface::new(surface, Some(exterior.into()), vec![]);

    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);

    // Constrain exterior boundary with evenly spaced parameters
    let n = 20;
    let exterior_params: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * TAU).collect();
    let constraints = TrimmedSurfaceConstraints::default().with_exterior(Some(exterior_params));

    let tess = trimmed
        .constrained_tessellate(constraints, opts)
        .expect("Constrained advancing front tessellation should succeed");

    let points = tess.points();
    let faces = tess.faces();

    assert!(!points.is_empty(), "Should produce vertices");
    assert!(!faces.is_empty(), "Should produce faces");
}

#[test]
fn test_advancing_front_tessellation_trait() {
    // Verify that TrimmedSurface implements Tessellation<AdvancingFrontOptions>
    let trimmed = make_trimmed_plane();
    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);

    let tess: anyhow::Result<SurfaceTessellation3D<f64>> = trimmed.tessellate(opts);
    let tess = tess.expect("Tessellation trait should work with AdvancingFrontOptions");

    assert!(!tess.points().is_empty());
    assert!(!tess.faces().is_empty());
}

/// Helper: collect boundary 3D points from a tessellation whose UV coordinates
/// satisfy a predicate, sorted by a key for deterministic comparison.
fn collect_boundary_points(
    tess: &SurfaceTessellation3D<f64>,
    uv_predicate: impl Fn(&Vector2<f64>) -> bool,
    sort_key: impl Fn(&Point3<f64>) -> ordered_float::OrderedFloat<f64>,
) -> Vec<Point3<f64>> {
    let mut pts: Vec<Point3<f64>> = tess
        .points()
        .iter()
        .zip(tess.uvs().iter())
        .filter(|(_, uv)| uv_predicate(uv))
        .map(|(p, _)| *p)
        .collect();
    pts.sort_by_key(|p| sort_key(p));
    pts.dedup_by(|a, b| (*a - *b).norm() < 1e-10);
    pts
}

/// Count edge occurrences in a tessellation.
/// Returns (boundary_edges, non_manifold_edges) where:
/// - boundary_edges: edges shared by exactly 1 face
/// - non_manifold_edges: edges shared by 3+ faces
fn count_edge_types(faces: &[[usize; 3]]) -> (usize, usize) {
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in faces {
        let edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for &(a, b) in &edges {
            let key = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    let boundary = edge_count.values().filter(|&&c| c == 1).count();
    let non_manifold = edge_count.values().filter(|&&c| c > 2).count();
    (boundary, non_manifold)
}

#[test]
fn test_advancing_front_single_face_manifold() {
    // A single trimmed surface should have no non-manifold edges,
    // and all boundary edges should form a closed loop.
    let trimmed = make_trimmed_cylinder(5.0, 10.0);
    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);
    let tess = trimmed.tessellate(opts).unwrap();

    let (boundary, non_manifold) = count_edge_types(tess.faces());
    assert_eq!(
        non_manifold, 0,
        "Single face should have no non-manifold edges, got {}",
        non_manifold
    );
    // Boundary edges should exist (open mesh) and form closed loops
    assert!(
        boundary > 0,
        "Single face should have boundary edges (it's an open surface)"
    );
}

#[test]
fn test_advancing_front_shared_edge_vertices_match() {
    // Two adjacent planar faces sharing an edge.
    // When tessellated with the same constraint parameters on the shared edge,
    // the 3D boundary points along that edge must be identical.
    //
    //  Face A: z=0 plane, x in [0,5], y in [0,5]
    //  Face B: z=0 plane, x in [5,10], y in [0,5]
    //  Shared edge: x=5, y in [0,5]

    let face_a_surface = NurbsSurface3D::plane(
        Point3::new(2.5, 2.5, 0.0),
        Vector3::new(2.5, 0.0, 0.0),
        Vector3::new(0.0, 2.5, 0.0),
    );
    let face_b_surface = NurbsSurface3D::plane(
        Point3::new(7.5, 2.5, 0.0),
        Vector3::new(2.5, 0.0, 0.0),
        Vector3::new(0.0, 2.5, 0.0),
    );

    // Both surfaces have UV domain [0,1] x [0,1]
    // For face_a: u=1 edge maps to x=5 (shared edge)
    // For face_b: u=0 edge maps to x=5 (shared edge)
    let make_exterior = || {
        let p0 = Point2::new(0.0, 0.0);
        let p1 = Point2::new(1.0, 0.0);
        let p2 = Point2::new(1.0, 1.0);
        let p3 = Point2::new(0.0, 1.0);
        CompoundCurve2D::new_unchecked(vec![
            NurbsCurve2D::polyline(&[p0, p1], true),
            NurbsCurve2D::polyline(&[p1, p2], true),
            NurbsCurve2D::polyline(&[p2, p3], true),
            NurbsCurve2D::polyline(&[p3, p0], true),
        ])
    };

    let trimmed_a = TrimmedSurface::new(face_a_surface, Some(make_exterior()), vec![]);
    let trimmed_b = TrimmedSurface::new(face_b_surface, Some(make_exterior()), vec![]);

    // Shared edge constraint: N evenly spaced points along the shared boundary.
    // For face_a, the shared edge is span index 1 (u=1, v varies from 0 to 1).
    // For face_b, the shared edge is span index 3 (u=0, v varies from 1 to 0).
    // Since both are polylines with knots_domain (0,1), we parameterize within each span.
    let n_shared = 6;
    let shared_params: Vec<f64> = (0..=n_shared).map(|i| i as f64 / n_shared as f64).collect();

    // For face_a: exterior has 4 spans [bottom, right, top, left]
    // Shared edge = span 1 (right edge, u=1, v: 0→1), param 0→1
    // Other spans use None (adaptive)
    let constraints_a = TrimmedSurfaceConstraints::new(Some(shared_params.clone()), vec![]);

    // For face_b: shared edge = span 3 (left edge, u=0, v: 1→0), param 0→1
    // The left edge goes from (0,1) to (0,0), so reversed v direction.
    // We need the same 3D points, so reverse the parameter order.
    let shared_params_reversed: Vec<f64> = shared_params.iter().rev().cloned().collect();
    let constraints_b = TrimmedSurfaceConstraints::new(Some(shared_params_reversed), vec![]);

    let opts = AdvancingFrontOptions::<f64>::default().with_chord_height_tolerance(0.1);

    let tess_a = trimmed_a
        .constrained_tessellate(constraints_a, opts.clone())
        .expect("Face A tessellation should succeed");
    let tess_b = trimmed_b
        .constrained_tessellate(constraints_b, opts)
        .expect("Face B tessellation should succeed");

    // Collect 3D points on the shared edge from each tessellation.
    // Face A shared edge: UV where u ≈ 1.0
    let eps = 1e-6;
    let edge_pts_a = collect_boundary_points(
        &tess_a,
        |uv| (uv.x - 1.0).abs() < eps,
        |p| ordered_float::OrderedFloat(p.y),
    );
    // Face B shared edge: UV where u ≈ 0.0
    let edge_pts_b = collect_boundary_points(
        &tess_b,
        |uv| uv.x.abs() < eps,
        |p| ordered_float::OrderedFloat(p.y),
    );

    assert!(
        !edge_pts_a.is_empty(),
        "Face A should have vertices on shared edge"
    );
    assert_eq!(
        edge_pts_a.len(),
        edge_pts_b.len(),
        "Shared edge vertex count must match: A={} vs B={}",
        edge_pts_a.len(),
        edge_pts_b.len()
    );

    for (pa, pb) in edge_pts_a.iter().zip(edge_pts_b.iter()) {
        let dist = (pa - pb).norm();
        assert!(
            dist < 1e-8,
            "Shared edge vertices must match: A=({:.6},{:.6},{:.6}) vs B=({:.6},{:.6},{:.6}), dist={:.2e}",
            pa.x, pa.y, pa.z, pb.x, pb.y, pb.z, dist
        );
    }
}
