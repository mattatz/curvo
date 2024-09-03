use operation::BooleanOperation;

pub mod curve;
mod degeneracies;
pub mod node;
pub mod operation;
pub mod status;
mod vertex;

/// A trait for boolean operations.
pub trait Boolean<T> {
    type Output;
    type Option;

    fn union(&self, other: T, option: Self::Option) -> Self::Output;
    fn intersection(&self, other: T, option: Self::Option) -> Self::Output;
    fn difference(&self, other: T, option: Self::Option) -> Self::Output;
    fn boolean(&self, operation: BooleanOperation, other: T, option: Self::Option) -> Self::Output;
}

#[cfg(test)]
mod tests;
