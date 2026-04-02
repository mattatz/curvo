use crate::misc::FloatingPoint;

/// Options for advancing front tessellation of trimmed surfaces.
#[derive(Debug, Clone)]
pub struct AdvancingFrontOptions<T: FloatingPoint> {
    /// Tolerance: maximum allowed distance between the mesh and the surface.
    /// Controls triangle density based on surface curvature.
    pub tolerance: T,
    /// Minimum allowed 3D edge length (to prevent degenerate triangles near poles).
    pub min_edge_length: T,
    /// Maximum allowed 3D edge length (to cap triangle size on flat regions).
    pub max_edge_length: T,
}

impl<T: FloatingPoint> Default for AdvancingFrontOptions<T> {
    fn default() -> Self {
        Self {
            tolerance: T::from_f64(0.1).unwrap(),
            min_edge_length: T::from_f64(1e-6).unwrap(),
            max_edge_length: T::from_f64(1e3).unwrap(),
        }
    }
}

impl<T: FloatingPoint> AdvancingFrontOptions<T> {
    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Backward-compatible alias for `with_tolerance`.
    pub fn with_deflection(self, deflection: T) -> Self {
        self.with_tolerance(deflection)
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
