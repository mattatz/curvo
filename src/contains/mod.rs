pub mod compound_curve_contains_compound_curve;
pub mod compound_curve_contains_curve;
pub mod compound_curve_contains_point;
pub mod curve_contains_compound_curve;
pub mod curve_contains_curve;
pub mod curve_contains_point;

pub use curve_contains_point::*;

/// Trait for determining if a target is inside a geometry.
pub trait Contains<T> {
    type Option;
    fn contains(&self, target: &T, option: Self::Option) -> anyhow::Result<bool>;
}

#[cfg(test)]
mod tests;
