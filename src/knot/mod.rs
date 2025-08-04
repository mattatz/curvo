pub mod knot_multiplicity;
pub mod knot_vector;
pub use knot_multiplicity::*;
pub use knot_vector::*;

/// Side of a knot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnotSide {
    None,
    Below,
    Above,
}

impl Default for KnotSide {
    fn default() -> Self {
        KnotSide::None
    }
}
