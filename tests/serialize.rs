use approx::assert_relative_eq;
use curvo::prelude::{NurbsCurve3D, NurbsSurface, NurbsSurface3D};
use nalgebra::{Point3, Vector3};

#[test]
fn test_curve_serialization() {
    let curve =
        NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    let json = serde_json::to_string_pretty(&curve).unwrap();
    // println!("{}", json);

    let der: NurbsCurve3D<f64> = serde_json::from_str(&json).unwrap();
    assert_relative_eq!(
        curve
            .control_points_iter()
            .flat_map(|p| p.coords.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
        der.control_points_iter()
            .flat_map(|p| p.coords.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(curve.degree(), der.degree());
    assert_relative_eq!(curve.knots().as_slice(), der.knots().as_slice());
}

#[test]
fn test_surface_serialization() {
    let curve =
        NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    let surface = NurbsSurface::extrude(&curve, &Vector3::z());
    let json = serde_json::to_string_pretty(&surface).unwrap();
    let der: NurbsSurface3D<f64> = serde_json::from_str(&json).unwrap();
    // println!("{}", json);

    assert_relative_eq!(
        surface
            .control_points()
            .iter()
            .flat_map(|column| { column.iter().flat_map(|p| p.coords.as_slice()) })
            .collect::<Vec<_>>()
            .as_slice(),
        der.control_points()
            .iter()
            .flat_map(|column| { column.iter().flat_map(|p| p.coords.as_slice()) })
            .collect::<Vec<_>>()
            .as_slice()
    );
    assert_eq!(surface.u_degree(), der.u_degree());
    assert_eq!(surface.v_degree(), der.v_degree());
    assert_relative_eq!(surface.u_knots().as_slice(), der.u_knots().as_slice());
    assert_relative_eq!(surface.v_knots().as_slice(), der.v_knots().as_slice());
}
