use curvo::prelude::*;
use nalgebra::{Point3, Translation3, Vector3};

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
    let opts = AdvancingFrontOptions::<f64>::default().with_deflection(0.1);
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
    let opts = AdvancingFrontOptions::<f64>::default().with_deflection(0.1);
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
    let opts_coarse = AdvancingFrontOptions::<f64>::default().with_deflection(0.5);
    let mesher_coarse = AdvancingFrontMesher::new(&trimmed, opts_coarse);
    let tess_coarse = mesher_coarse.mesh().unwrap();

    // Fine tessellation
    let opts_fine = AdvancingFrontOptions::<f64>::default().with_deflection(0.05);
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
    let opts = AdvancingFrontOptions::<f64>::default().with_deflection(0.1);
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
