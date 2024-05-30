/// Trait for types that can be inverted.
pub trait Invertible: Clone {
    fn invert(&mut self);
    fn inverse(&self) -> Self {
        let mut inv = self.clone();
        inv.invert();
        inv
    }
}
