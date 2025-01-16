use crate::misc::FloatingPoint;

/// A struct representing constraints for a trimmed surface tessellation
#[derive(Clone, Debug, Default)]
pub struct TrimmedSurfaceConstraints<T: FloatingPoint> {
    exterior: Option<Vec<T>>,
    interiors: Vec<Option<Vec<T>>>,
}

impl<T: FloatingPoint> TrimmedSurfaceConstraints<T> {
    pub fn new(exterior: Option<Vec<T>>, interiors: Vec<Option<Vec<T>>>) -> Self {
        Self {
            exterior,
            interiors,
        }
    }

    pub fn with_exterior(mut self, exterior: Option<Vec<T>>) -> Self {
        self.exterior = exterior;
        self
    }

    pub fn with_interiors(mut self, interiors: Vec<Option<Vec<T>>>) -> Self {
        self.interiors = interiors;
        self
    }

    pub fn push_interior(&mut self, interior: Option<Vec<T>>) {
        self.interiors.push(interior);
    }

    pub fn exterior(&self) -> Option<&Vec<T>> {
        self.exterior.as_ref()
    }

    pub fn interiors(&self) -> &Vec<Option<Vec<T>>> {
        &self.interiors
    }
}
