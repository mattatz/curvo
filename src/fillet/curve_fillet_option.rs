use crate::misc::FloatingPoint;

/// Fillet the sharp corners of the curve with a given radius
#[derive(Debug, Clone, Copy)]
pub struct FilletRadiusOption<T: FloatingPoint> {
    radius: T,
}

impl<T: FloatingPoint> FilletRadiusOption<T> {
    pub fn new(radius: T) -> Self {
        Self { radius }
    }

    pub fn radius(&self) -> T {
        self.radius
    }
}

/// Only fillet the sharp corners at the specified parameter positions with a given radius
#[derive(Debug, Clone)]
pub struct FilletRadiusParameterOption<T: FloatingPoint> {
    radius: T,
    parameters: Vec<T>,
}

impl<T: FloatingPoint> FilletRadiusParameterOption<T> {
    pub fn new(radius: T, parameters: Vec<T>) -> Self {
        Self { radius, parameters }
    }

    pub fn from_single(radius: T, parameter: T) -> Self {
        Self {
            radius,
            parameters: vec![parameter],
        }
    }

    pub fn radius(&self) -> T {
        self.radius
    }

    pub fn parameters(&self) -> &[T] {
        &self.parameters
    }

    pub fn with_parameters(&mut self, parameters: Vec<T>) -> &mut Self {
        self.parameters = parameters;
        self
    }

    pub fn add_parameter(&mut self, parameter: T) -> &mut Self {
        self.parameters.push(parameter);
        self
    }

    pub fn with_radius(&mut self, radius: T) -> &mut Self {
        self.radius = radius;
        self
    }
}

/// Fillet the sharp corners of the curve with a given distance
#[derive(Debug, Clone, Copy)]
pub struct FilletDistanceOption<T: FloatingPoint> {
    distance: T,
}

impl<T: FloatingPoint> FilletDistanceOption<T> {
    pub fn new(distance: T) -> Self {
        Self { distance }
    }

    pub fn distance(&self) -> T {
        self.distance
    }
}
