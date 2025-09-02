use std::marker::PhantomData;

use nalgebra::{RealField, U4};

use crate::tessellation::DefaultDivider;

/// Options for adaptive tessellation of a surface
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveTessellationOptions<T = f64, D = U4, F = DefaultDivider<T, D>> {
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
    /// Divider function
    pub divider: Option<F>,
    _marker: PhantomData<D>,
}

impl<T: RealField, D, F> Default for AdaptiveTessellationOptions<T, D, F> {
    fn default() -> Self {
        Self {
            norm_tolerance: T::from_f64(2.5e-2).unwrap(),
            min_divs_u: 1,
            min_divs_v: 1,
            min_depth: 0,
            max_depth: 8,
            divider: None,
            _marker: PhantomData,
        }
    }
}

impl<T: RealField, D, F> AdaptiveTessellationOptions<T, D, F> {
    /// Set the tolerance for the normal vector
    pub fn with_norm_tolerance(mut self, norm_tolerance: T) -> Self {
        self.norm_tolerance = norm_tolerance;
        self
    }

    /// Set the minimum number of divisions in u direction
    pub fn with_min_divs_u(mut self, min_divs_u: usize) -> Self {
        self.min_divs_u = min_divs_u;
        self
    }

    /// Set the minimum number of divisions in v direction
    pub fn with_min_divs_v(mut self, min_divs_v: usize) -> Self {
        self.min_divs_v = min_divs_v;
        self
    }

    /// Set the minimum depth for division
    pub fn with_min_depth(mut self, min_depth: usize) -> Self {
        self.min_depth = min_depth;
        self
    }

    /// Set the maximum depth for division
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set the divider function
    pub fn with_divider(mut self, divider: Option<F>) -> Self {
        self.divider = divider;
        self
    }
}
