use crate::misc::FloatingPoint;

/// Options for advancing front tessellation of trimmed surfaces.
#[derive(Debug, Clone)]
pub struct AdvancingFrontOptions<T: FloatingPoint> {
    /// Maximum allowed chord height (= distance between the mesh and the
    /// underlying surface). Controls triangle density based on surface
    /// curvature.
    pub chord_height_tolerance: T,
    /// Minimum allowed 3D edge length (to prevent degenerate triangles near poles).
    pub min_edge_length: T,
    /// Maximum allowed 3D edge length (to cap triangle size on flat regions).
    pub max_edge_length: T,
}

impl<T: FloatingPoint> Default for AdvancingFrontOptions<T> {
    fn default() -> Self {
        Self {
            chord_height_tolerance: T::from_f64(0.1).unwrap(),
            min_edge_length: T::from_f64(1e-6).unwrap(),
            max_edge_length: T::from_f64(1e3).unwrap(),
        }
    }
}

impl<T: FloatingPoint> AdvancingFrontOptions<T> {
    pub fn with_chord_height_tolerance(mut self, tolerance: T) -> Self {
        self.chord_height_tolerance = tolerance;
        self
    }

    pub fn with_min_edge_length(mut self, len: T) -> Self {
        self.min_edge_length = len;
        self
    }

    pub fn with_max_edge_length(mut self, len: T) -> Self {
        self.max_edge_length = len;
        self
    }
}
