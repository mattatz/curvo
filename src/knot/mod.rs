pub mod knot_multiplicity;
pub mod knot_vector;
pub use knot_multiplicity::*;
pub use knot_vector::*;

/// Side of a knot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum KnotSide {
    #[default]
    None,
    Below,
    Above,
}
