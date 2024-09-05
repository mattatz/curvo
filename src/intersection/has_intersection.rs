///
pub trait HasIntersection<V, T>: HasIntersectionParameter<T> {
    fn a(&self) -> &V;
    fn b(&self) -> &V;
}

///
pub trait HasIntersectionParameter<T> {
    fn a_parameter(&self) -> T;
    fn b_parameter(&self) -> T;
}
