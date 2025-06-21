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

/// Only fillet the sharp corner at the specified parameter position with a given radius
#[derive(Debug, Clone, Copy)]
pub struct FilletRadiusParameterOption<T: FloatingPoint> {
    radius: T,
    parameter: T,
}

impl<T: FloatingPoint> FilletRadiusParameterOption<T> {
    pub fn new(radius: T, parameter: T) -> Self {
        Self { radius, parameter }
    }

    pub fn radius(&self) -> T {
        self.radius
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }

    pub fn with_parameter(&mut self, parameter: T) -> &mut Self {
        self.parameter = parameter;
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
