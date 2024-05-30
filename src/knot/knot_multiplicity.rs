/// A struct that represents the multiplicity of a knot.
#[derive(Clone, Debug)]
pub struct KnotMultiplicity<T> {
    knot: T,
    multiplicity: usize,
}

impl<T> KnotMultiplicity<T> {
    pub fn new(knot: T, multiplicity: usize) -> Self {
        Self { knot, multiplicity }
    }

    pub fn knot(&self) -> &T {
        &self.knot
    }

    pub fn multiplicity(&self) -> usize {
        self.multiplicity
    }

    pub fn increment_multiplicity(&mut self) {
        self.multiplicity += 1;
    }
}
