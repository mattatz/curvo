use operation::BooleanOperation;

pub mod boolean_compound_curve;
pub mod boolean_curve;
pub mod boolean_region;
mod clip;
mod degeneracies;
mod has_parameter;
pub mod node;
pub mod operation;
pub mod status;
mod vertex;
pub use clip::Clip;

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;
    type Option;

    fn union(&self, other: T, option: Self::Option) -> Self::Output;

    fn intersection(&self, other: T, option: Self::Option) -> Self::Output;

    fn difference(&self, other: T, option: Self::Option) -> Self::Output;

    fn boolean(&self, operation: BooleanOperation, other: T, option: Self::Option) -> Self::Output {
        match operation {
            BooleanOperation::Union => self.union(other, option),
            BooleanOperation::Intersection => self.intersection(other, option),
            BooleanOperation::Difference => self.difference(other, option),
        }
    }
}

#[cfg(test)]
mod tests;
