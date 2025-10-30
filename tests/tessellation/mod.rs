#![allow(unused_imports)]

use approx::assert_relative_eq;
use curvo::prelude::{
    AdaptiveTessellationOptions, SurfaceTessellation, Tessellation, TrimmedSurface,
};
use nalgebra::U4;
use std::path::Path;

#[test]
#[cfg(all(feature = "serde", feature = "approx"))]
fn test_union_case() {
    let s = include_str!("./union.json");
    let trimmed: TrimmedSurface<f64> = serde_json::from_str(s).unwrap();
    let option = AdaptiveTessellationOptions::<_>::default().with_norm_tolerance(1e-2);
    let tess: SurfaceTessellation<f64, U4> = trimmed.tessellate(Some(option)).unwrap();

    let expected_json = include_str!("./union_tessellation.json");
    let expected: SurfaceTessellation<f64, U4> = serde_json::from_str(expected_json).unwrap();

    // Verify tessellation matches expected result
    assert_relative_eq!(tess, expected, epsilon = 1e-10);

    /*
    // Write tessellation result for debugging
    let rel: &'static str = file!();
    let dir = Path::new(rel).parent().unwrap();
    let path = dir.join("union_tessellation.json");
    std::fs::write(path, serde_json::to_string_pretty(&tess).unwrap()).unwrap();
    */
}
