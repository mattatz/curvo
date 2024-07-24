use approx::assert_relative_eq;
use curvo::prelude::NurbsCurve3D;
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
