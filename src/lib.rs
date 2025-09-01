#![allow(clippy::needless_range_loop)]
#![allow(clippy::needless_doctest_main)]
//! Curvo is a NURBS modeling library for Rust.
//!
//! This library enables not only the creation of NURBS curves from control points, knot vectors, and weights associated with each control point, but also supports generating curves that precisely pass through the given control points and creating periodic curves. Additionally, it allows for the construction of NURBS surfaces through operations such as _extruding_ and _lofting_ based on NURBS curves as inputs.
//!
//! The modeling operations for NURBS surfaces supported by this library currently include the following:
//! - Extrude
//! - Loft
//! - Sweep
//! - Revolve
//!
//! ## Example
//!
//! ```
//! use curvo::prelude::*;
//! use nalgebra::{Point3, Rotation3, Translation3, Vector3};
//! use std::f64::consts::FRAC_PI_2;
//!
//! fn main() {
//!     // Create a set of points to interpolate
//!     let points = vec![
//!         Point3::new(-1.0, -1.0, 0.),
//!         Point3::new(1.0, -1.0, 0.),
//!         Point3::new(1.0, 1.0, 0.),
//!         Point3::new(-1.0, 1.0, 0.),
//!         Point3::new(-1.0, 2.0, 0.),
//!         Point3::new(1.0, 2.5, 0.),
//!     ];
//!
//!     // Create a NURBS curve that interpolates the given points with degree 3
//!     // You can also specify the precision of the curve by generic type (f32 or f64)
//!     let interpolated = NurbsCurve3D::<f64>::try_interpolate(&points, 3).unwrap();
//!
//!     // NURBS curve & surface can be transformed by nalgebra's matrix
//!     let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
//!     let translation = Translation3::new(0., 0., 3.);
//!     let transform_matrix = translation * rotation; // nalgebra::Isometry3
//!
//!     // Transform the curve by the given matrix (nalgebra::Isometry3 into nalgebra::Matrix4)
//!     let offsetted = interpolated.transformed(&transform_matrix.into());
//!
//!     // Create a NURBS surface by lofting two NURBS curves
//!     let lofted = NurbsSurface::try_loft(
//!         &[interpolated, offsetted],
//!         Some(3), // degree of v direction
//!     ).unwrap();
//!
//!     // Tessellate the surface in adaptive manner about curvature for efficient rendering
//!     let option = AdaptiveTessellationOptions::<_> {
//!         norm_tolerance: 1e-4,
//!         ..Default::default()
//!     };
//!     let tessellation = lofted.tessellate(Some(option));
//! }
//! ```

mod boolean;
mod bounding_box;
mod closest_parameter;
mod contains;
mod curve;
mod decompose;
mod dimension;
mod discontinuity;
mod fillet;
mod intersects;
mod knot;
mod misc;
mod offset;
mod polygon_mesh;
mod region;
mod split;
mod surface;
mod tessellation;
mod trim;
use closest_parameter::*;

pub mod prelude {
    pub use crate::boolean::*;
    pub use crate::bounding_box::*;
    pub use crate::contains::*;
    pub use crate::curve::*;
    pub use crate::decompose::*;
    pub use crate::dimension::*;
    pub use crate::discontinuity::*;
    pub use crate::fillet::*;
    pub use crate::intersects::*;
    pub use crate::knot::*;
    pub use crate::misc::{
        binomial::*, curvature::*, end_points::*, floating_point::*, frenet_frame::*,
        invertible::*, line::*, orientation::*, plane::*, polygon_boundary::*, ray::*,
        transformable::*, transpose::*, trigonometry::*,
    };
    pub use crate::offset::*;
    pub use crate::polygon_mesh::*;
    pub use crate::region::*;
    pub use crate::split::*;
    pub use crate::surface::*;
    pub use crate::tessellation::{
        adaptive_tessellation_node::AdaptiveTessellationNode,
        adaptive_tessellation_option::AdaptiveTessellationOptions,
        boundary_constraints::BoundaryConstraints, surface_tessellation::*,
        trimmed_surface_constraints::TrimmedSurfaceConstraints, ConstrainedTessellation,
        DefaultDivider, DividableDirection, Tessellation,
    };
    pub use crate::trim::*;
}
