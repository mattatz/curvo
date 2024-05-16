/// A trait for objects that can be transformed by a given type.
pub trait Transformable<T>: Clone {
    fn transform(&mut self, transform: T);

    fn transformed(&self, transform: T) -> Self {
        let mut clone = self.clone();
        clone.transform(transform);
        clone
    }
}
