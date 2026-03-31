use crate::misc::FloatingPoint;

/// Options for advancing front tessellation of trimmed surfaces.
#[derive(Debug, Clone)]
pub struct AdvancingFrontOptions<T: FloatingPoint> {
    /// Deflection tolerance: maximum allowed distance between the mesh and the surface.
    /// Controls triangle density based on surface curvature.
    pub deflection: T,
    /// Minimum allowed 3D edge length (to prevent degenerate triangles near poles).
    pub min_edge_length: T,
    /// Maximum allowed 3D edge length (to cap triangle size on flat regions).
    pub max_edge_length: T,
    /// Minimum angle in radians for generated triangles.
    pub min_angle: T,
}

impl<T: FloatingPoint> Default for AdvancingFrontOptions<T> {
    fn default() -> Self {
        Self {
            deflection: T::from_f64(0.1).unwrap(),
            min_edge_length: T::from_f64(1e-6).unwrap(),
            max_edge_length: T::from_f64(1e3).unwrap(),
            min_angle: T::from_f64(std::f64::consts::FRAC_PI_6).unwrap(), // 30°
        }
    }
}

impl<T: FloatingPoint> AdvancingFrontOptions<T> {
    pub fn with_deflection(mut self, deflection: T) -> Self {
        self.deflection = deflection;
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

    pub fn with_min_angle(mut self, angle: T) -> Self {
        self.min_angle = angle;
        self
    }
}
