use crate::misc::FloatingPoint;

/// A struct representing constraints for a trimmed surface tessellation
#[derive(Clone, Debug)]
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

    pub fn exterior(&self) -> Option<&Vec<T>> {
        self.exterior.as_ref()
    }

    pub fn interiors(&self) -> &Vec<Option<Vec<T>>> {
        &self.interiors
    }
}
