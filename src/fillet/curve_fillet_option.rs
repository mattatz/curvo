use crate::misc::FloatingPoint;

/// Fillet the sharp corners of the curve with a given radius
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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

impl<T: FloatingPoint> From<FilletRadiusParameterOption<T>> for FilletRadiusParameterSetOption<T> {
    /// Convert a `FilletRadiusParameterOption` to a `FilletRadiusParameterSetOption`
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use approx::assert_relative_eq;
    ///
    /// let option = FilletRadiusParameterOption::<f64>::new(1.0, vec![0.0, 0.5, 1.0]);
    /// let set = FilletRadiusParameterSetOption::from(option);
    /// assert_eq!(set, FilletRadiusParameterSetOption::<f64>::new(vec![
    ///     FilletRadiusParameterSet::new(1.0, 0.0),
    ///     FilletRadiusParameterSet::new(1.0, 0.5),
    ///     FilletRadiusParameterSet::new(1.0, 1.0),
    /// ]));
    /// ```
    fn from(option: FilletRadiusParameterOption<T>) -> Self {
        let radius = option.radius();
        Self::new(
            option
                .parameters()
                .iter()
                .map(|&p| FilletRadiusParameterSet::new(radius, p))
                .collect(),
        )
    }
}

/// Only fillet the sharp corners with the specified parameter position & radius sets
#[derive(Debug, Clone, PartialEq)]
pub struct FilletRadiusParameterSetOption<T: FloatingPoint> {
    radius_parameter_sets: Vec<FilletRadiusParameterSet<T>>,
}

impl<T: FloatingPoint> FilletRadiusParameterSetOption<T> {
    pub fn new(radius_parameter_sets: Vec<FilletRadiusParameterSet<T>>) -> Self {
        Self {
            radius_parameter_sets,
        }
    }

    pub fn radius_parameter_sets(&self) -> &[FilletRadiusParameterSet<T>] {
        &self.radius_parameter_sets
    }

    pub fn with_radius_parameter_sets(
        &mut self,
        radius_parameter_sets: Vec<FilletRadiusParameterSet<T>>,
    ) -> &mut Self {
        self.radius_parameter_sets = radius_parameter_sets;
        self
    }

    pub fn add_radius_parameter_set(
        &mut self,
        radius_parameter_set: FilletRadiusParameterSet<T>,
    ) -> &mut Self {
        self.radius_parameter_sets.push(radius_parameter_set);
        self
    }
}

/// A set of radius and parameter for filletting a sharp corner
#[derive(Debug, Clone, PartialEq)]
pub struct FilletRadiusParameterSet<T: FloatingPoint> {
    radius: T,
    parameter: T,
}

impl<T: FloatingPoint> FilletRadiusParameterSet<T> {
    pub fn new(radius: T, parameter: T) -> Self {
        Self { radius, parameter }
    }

    pub fn radius(&self) -> T {
        self.radius
    }

    pub fn parameter(&self) -> T {
        self.parameter
    }

    pub fn with_radius(&mut self, radius: T) -> &mut Self {
        self.radius = radius;
        self
    }

    pub fn with_parameter(&mut self, parameter: T) -> &mut Self {
        self.parameter = parameter;
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
