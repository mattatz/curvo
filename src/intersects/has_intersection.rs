/// A trait for types that have an intersection.
pub trait HasIntersection<VA, VB, A, B>: HasIntersectionParameter<A, B> {
    fn a(&self) -> &VA;
    fn b(&self) -> &VB;
}

/// A trait for types that have intersection parameters.
pub trait HasIntersectionParameter<A, B> {
    fn a_parameter(&self) -> A;
    fn b_parameter(&self) -> B;
}
