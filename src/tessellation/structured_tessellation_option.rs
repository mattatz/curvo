use nalgebra::RealField;

/// Options for structured (tensor-product) surface tessellation.
///
/// Samples u and v once each and takes their tensor product: regular
/// connectivity, no T-junctions (unlike the adaptive quad-tree). Grid lines are
/// still placed adaptively per direction — each isocurve is flattened to
/// `tolerance`. "structured" refers to the connectivity, not to uniform spacing.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuredTessellationOptions<T> {
    /// Flatness tolerance (squared world distance) for per-direction sampling.
    pub tolerance: T,
}

impl<T: RealField> Default for StructuredTessellationOptions<T> {
    fn default() -> Self {
        Self {
            tolerance: T::from_f64(1e-2).unwrap(),
        }
    }
}

impl<T: RealField> StructuredTessellationOptions<T> {
    pub fn new(tolerance: T) -> Self {
        Self { tolerance }
    }

    pub fn with_tolerance(mut self, tolerance: T) -> Self {
        self.tolerance = tolerance;
        self
    }
}
