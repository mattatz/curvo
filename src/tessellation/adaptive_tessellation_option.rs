use std::marker::PhantomData;

use nalgebra::{DimName, RealField};

use crate::tessellation::{
    adaptive_tessellation_node::AdaptiveTessellationNode, DefaultDivider, DividableDirection
};

/// Options for adaptive tessellation of a surface
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveTessellationOptions<
    T: RealField,
    D: DimName,
    F: Fn(&AdaptiveTessellationNode<T, D>) -> Option<DividableDirection> = DefaultDivider<T, D>,
> {
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
    pub _marker: PhantomData<D>,
}

impl<
        T: RealField,
        D: DimName,
        F: Fn(&AdaptiveTessellationNode<T, D>) -> Option<DividableDirection>,
    > Default for AdaptiveTessellationOptions<T, D, F>
{
    fn default() -> Self {
        Self {
            norm_tolerance: T::from_f64(2.5e-2).unwrap(),
            min_divs_u: 1,
            min_divs_v: 1,
            min_depth: 0,
            max_depth: 8,
            divider: None::<F>,
            _marker: PhantomData,
        }
    }
}
