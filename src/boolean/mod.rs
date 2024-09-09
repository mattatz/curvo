use operation::BooleanOperation;

mod clip;
pub mod compound_curve;
pub mod curve;
mod degeneracies;
mod has_parameter;
pub mod node;
pub mod operation;
pub mod status;
mod vertex;
pub use clip::Clipped;

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;
    type Option;

    fn union(&self, other: T, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Union, other, option)
    }

    fn intersection(&self, other: T, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Intersection, other, option)
    }

    fn difference(&self, other: T, option: Self::Option) -> Self::Output {
        self.boolean(BooleanOperation::Difference, other, option)
    }

    fn boolean(&self, operation: BooleanOperation, other: T, option: Self::Option) -> Self::Output;
}

#[cfg(test)]
mod tests;
