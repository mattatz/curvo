use curvo::prelude::NurbsCurve3D;
use nalgebra::{Point3, Vector3};
use serde::Serialize;

#[test]
fn test_serialization() {
    let curve =
        NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    let json = serde_json::to_string_pretty(&curve).unwrap();
    println!("{}", json);
}
