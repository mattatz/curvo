# Curvo

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/mattatz/curvo#license)
[![Crates.io](https://img.shields.io/crates/v/curvo.svg)](https://crates.io/crates/curvo)
[![Docs](https://docs.rs/curvo/badge.svg)](https://docs.rs/curvo/latest/curvo/)
[![Test](https://github.com/mattatz/curvo/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/mattatz/curvo/actions/workflows/test.yml)

Curvo is a NURBS curve / surface modeling library for Rust.

![Visualization on bevy](https://github.com/mattatz/curvo/assets/1085910/50b44a8c-d8c1-43e0-8db5-d6fff52300e6)
*Visualization on [Bevy](https://bevyengine.org/)*

This library enables not only the creation of NURBS curves from control points, knot vectors, and weights associated with each control point, but also supports generating curves that precisely pass through the given control points and creating periodic curves. Additionally, it allows for the construction of NURBS surfaces through operations such as _extruding_ and _lofting_ based on NURBS curves as inputs.

The modeling operations for NURBS surfaces supported by this library currently include the following:
- Extrude
- Loft
- Sweep
- Revolve

<img src="https://github.com/mattatz/curvo/assets/1085910/1eecea06-5848-48f3-9b7d-916715082f09" width="360px" alt="Sweep the profile curve along the rail curve to create a surface" />

<img width="360px" alt="Revolve the profile curve around an z-axis by PI radians to create a NURBS surface" src="https://github.com/mattatz/curvo/assets/1085910/3456dc46-9977-446e-8d5c-eafe109093a7">

The supported features also include finding the closest point on NURBS curves, finding intersections between two NURBS curves, and dividing based on arc length.

<img src="https://github.com/mattatz/curvo/assets/1085910/f07cf1c4-3994-44d1-95a6-86a8feaf3d2e" width="360px" alt="Find closest point on the NURBS curve" />

<img src="https://github.com/mattatz/curvo/assets/1085910/dd453a50-46ca-4279-b547-c1a667d885a9" width="360px" alt="Divide the NURBS curve based on arc length" />

<img src="https://github.com/mattatz/curvo/assets/1085910/8f29cb4b-d3d5-4553-9f41-deb8cc63b132" width="360px" alt="Find intersection points between two NURBS curves" />

## Additional features

<img src="https://github.com/user-attachments/assets/a4cef924-b3a9-410d-aebc-477f6f86f193" width="360px" alt="Boolean operations between NURBS curves" />

<img src="https://github.com/mattatz/curvo/assets/1085910/754d2a96-0f6b-40f7-b6b1-106789d585b7" width="360px" alt="Smooth periodic points interpolation with knot styles" />

<img src="https://github.com/mattatz/curvo/assets/1085910/87ebdb66-f0df-46cc-8cb4-01f597080ffb" width="360px" alt="Ellipse arc generation" />

<img src="https://github.com/mattatz/curvo/assets/1085910/55f214ec-668a-4d18-8fa8-bfa5ca23b3a2" width="360px" alt="Iso-curves on a surface" />

<img src="https://github.com/user-attachments/assets/3b739a9f-531c-4264-90ce-a65e426439b3" width="360px" alt="Closest point on surface" />

## Usage

```rust
// Create a set of points to interpolate
let points = vec![
    Point3::new(-1.0, -1.0, 0.),
    Point3::new(1.0, -1.0, 0.),
    Point3::new(1.0, 1.0, 0.),
    Point3::new(-1.0, 1.0, 0.),
    Point3::new(-1.0, 2.0, 0.),
    Point3::new(1.0, 2.5, 0.),
];

// Create a NURBS curve that interpolates the given points with degree 3
// You can also specify the precision of the curve by generic type (f32 or f64)
let interpolated = NurbsCurve3D::<f64>::try_interpolate(&points, 3).unwrap();

// NURBS curve & surface can be transformed by nalgebra's matrix
let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
let translation = Translation3::new(0., 0., 3.);
let transform_matrix = translation * rotation; // nalgebra::Isometry3

// Transform the curve by the given matrix (nalgebra::Isometry3 into nalgebra::Matrix4)
let offsetted = interpolated.transformed(&transform_matrix.into());

// Create a NURBS surface by lofting two NURBS curves
let lofted = NurbsSurface::try_loft(
  &[interpolated, offsetted],
  Some(3), // degree of v direction
).unwrap();

// Tessellate the surface in adaptive manner about curvature for efficient rendering
let option = AdaptiveTessellationOptions {
    norm_tolerance: 1e-4,
    ..Default::default()
};
let tessellation = lofted.tessellate(Some(option));

```

## Dependencies

- [nalgebra](https://crates.io/crates/nalgebra): this library heavily relies on nalgebra, a linear algebra library, to perform its computations.

## References

- [The NURBS Book](https://www.amazon.com/NURBS-Book-Monographs-Visual-Communication/dp/3540615458) by Piegl and Tiller

## Feature development sponsored by VUILD Inc.

The **NURBS boolean operations** feature in this project was developed at the request of [VUILD](https://vuild.co.jp/).  
They supported the development as a sponsor by funding the feature's implementation.
