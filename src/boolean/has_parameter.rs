/// A trait for types that have a parameter.
pub trait HasParameter<T> {
    fn parameter(&self) -> T;
}
