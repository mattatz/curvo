/// A struct that contains the parameter of a curve and the arc length of the curve at that parameter.
pub struct CurveLengthParameter<T: Copy> {
    parameter: T,
    length: T,
}

impl<T: Copy> CurveLengthParameter<T> {
    pub fn new(parameter: T, length: T) -> Self {
        Self { parameter, length }
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }

    pub fn length(&self) -> T {
        self.length
    }
}
