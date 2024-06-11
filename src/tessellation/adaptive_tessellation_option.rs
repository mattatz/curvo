use nalgebra::RealField;

/// Options for adaptive tessellation of a surface
#[derive(Clone, Debug)]
pub struct AdaptiveTessellationOptions<T: RealField> {
    /// Tolerance for the normal vector: if the L2 norm of the normal vectors is below this value, the edge is considered flat
    pub norm_tolerance: T,
    /// Minimum number of divisions in u direction
    pub min_divs_u: usize,
    /// Minimum number of divisions in v direction
    pub min_divs_v: usize,
    /// Minimum depth for division
    pub min_depth: usize,
    /// Maximum depth for division
    pub max_depth: usize,
}

impl<T: RealField> Default for AdaptiveTessellationOptions<T> {
    fn default() -> Self {
        Self {
            norm_tolerance: T::from_f64(2.5e-2).unwrap(),
            min_divs_u: 1,
            min_divs_v: 1,
            min_depth: 0,
            max_depth: 8,
        }
    }
}
