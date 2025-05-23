#![allow(unused_imports)]

use approx::assert_relative_eq;
use curvo::prelude::{FloatingPoint, NurbsCurve, NurbsCurve3D, NurbsSurface, NurbsSurface3D};
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, Point3, Vector3};

#[test]
#[cfg(feature = "serde")]
fn test_curve_serialization() {
    use curvo::prelude::CompoundCurve;
    use nalgebra::{U3, U4};

    let curve =
        NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    let json = serde_json::to_string_pretty(&curve).unwrap();
    // println!("{}", json);

    let der: NurbsCurve3D<f64> = serde_json::from_str(&json).unwrap();
    assert_curve_eq(&curve, &der);

    let compound = CompoundCurve::try_new(vec![curve.clone()]).unwrap();
    let json = serde_json::to_string_pretty(&compound).unwrap();
    let der: CompoundCurve<f64, U4> = serde_json::from_str(&json).unwrap();
    compound.spans().iter().enumerate().for_each(|(i, span)| {
        let o = &der.spans()[i];
        assert_relative_eq!(span.knots().as_slice(), o.knots().as_slice());
    });
}

#[test]
#[cfg(feature = "serde")]
fn test_surface_serialization() {
    let curve =
        NurbsCurve3D::try_circle(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.).unwrap();
    let surface = NurbsSurface::extrude(&curve, &Vector3::z());
    let json = serde_json::to_string_pretty(&surface).unwrap();
    let der: NurbsSurface3D<f64> = serde_json::from_str(&json).unwrap();
    // println!("{}", json);

    assert_surface_eq(&surface, &der);
}

#[test]
#[cfg(feature = "serde")]
fn test_trimmed_surface_serialization() {
    use curvo::prelude::{NurbsCurve2D, TrimmedSurface};
    use nalgebra::{Point2, Vector2};

    let profile = NurbsCurve3D::polyline(&[Point3::origin(), Point3::new(5., 0., 0.)], true);
    let plane_surface = NurbsSurface::extrude(&profile, &(Vector3::z() * 10.));
    let trimmed = TrimmedSurface::new(
        plane_surface.clone(),
        None,
        vec![
            NurbsCurve2D::try_circle(&Point2::new(0.5, 0.5), &Vector2::x(), &Vector2::y(), 0.25)
                .unwrap(),
        ],
    );
    let json = serde_json::to_string_pretty(&trimmed).unwrap();
    let der: TrimmedSurface<f64> = serde_json::from_str(&json).unwrap();

    assert_surface_eq(trimmed.surface(), der.surface());
    assert_eq!(trimmed.exterior().is_none(), der.exterior().is_none());

    trimmed
        .interiors()
        .iter()
        .zip(der.interiors())
        .for_each(|(a, b)| {
            assert_curve_eq(a, b);
        });
}

#[allow(dead_code)]
fn assert_curve_eq<T: FloatingPoint, D: DimName>(a: &NurbsCurve<T, D>, b: &NurbsCurve<T, D>)
where
    DefaultAllocator: Allocator<D>,
{
    assert_eq!(a.degree(), b.degree());
    assert_relative_eq!(a.knots().as_slice(), b.knots().as_slice());
    assert_relative_eq!(
        a.control_points_iter()
            .flat_map(|p| p.coords.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
        b.control_points_iter()
            .flat_map(|p| p.coords.as_slice())
            .collect::<Vec<_>>()
            .as_slice()
    );
}

#[allow(dead_code)]
fn assert_surface_eq<T: FloatingPoint>(a: &NurbsSurface3D<T>, b: &NurbsSurface3D<T>) {
    assert_eq!(a.u_degree(), b.u_degree());
    assert_eq!(a.v_degree(), b.v_degree());
    assert_relative_eq!(a.u_knots().as_slice(), b.u_knots().as_slice());
    assert_relative_eq!(a.v_knots().as_slice(), b.v_knots().as_slice());
    assert_relative_eq!(
        a.control_points()
            .iter()
            .flat_map(|column| { column.iter().flat_map(|p| p.coords.as_slice()) })
            .collect::<Vec<_>>()
            .as_slice(),
        b.control_points()
            .iter()
            .flat_map(|column| { column.iter().flat_map(|p| p.coords.as_slice()) })
            .collect::<Vec<_>>()
            .as_slice()
    );
}
