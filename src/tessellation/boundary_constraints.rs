use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, U1,
};

use crate::{misc::FloatingPoint, surface::NurbsSurface};

/// A struct representing constraints at the boundary of a surface tessellation
#[derive(Clone, Debug)]
pub struct BoundaryConstraints<T: FloatingPoint> {
    v_parameters_at_u_min: Option<Vec<T>>,
    u_parameters_at_v_min: Option<Vec<T>>,
    v_parameters_at_u_max: Option<Vec<T>>,
    u_parameters_at_v_max: Option<Vec<T>>,
}

impl<T: FloatingPoint> Default for BoundaryConstraints<T> {
    fn default() -> Self {
        Self {
            v_parameters_at_u_min: None,
            u_parameters_at_v_min: None,
            v_parameters_at_u_max: None,
            u_parameters_at_v_max: None,
        }
    }
}

impl<T: FloatingPoint> BoundaryConstraints<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_v_parameters_at_u_min(mut self, v_parameters: Vec<T>) -> Self {
        self.v_parameters_at_u_min = Some(v_parameters);
        self
    }

    pub fn with_u_parameters_at_v_min(mut self, u_parameters: Vec<T>) -> Self {
        self.u_parameters_at_v_min = Some(u_parameters);
        self
    }

    pub fn with_v_parameters_at_u_max(mut self, v_parameters: Vec<T>) -> Self {
        self.v_parameters_at_u_max = Some(v_parameters);
        self
    }

    pub fn with_u_parameters_at_v_max(mut self, u_parameters: Vec<T>) -> Self {
        self.u_parameters_at_v_max = Some(u_parameters);
        self
    }

    pub fn u_parameters_at_v_min(&self) -> Option<&Vec<T>> {
        self.u_parameters_at_v_min.as_ref()
    }

    pub fn u_parameters_at_v_max(&self) -> Option<&Vec<T>> {
        self.u_parameters_at_v_max.as_ref()
    }

    pub fn v_parameters_at_u_min(&self) -> Option<&Vec<T>> {
        self.v_parameters_at_u_min.as_ref()
    }

    pub fn v_parameters_at_u_max(&self) -> Option<&Vec<T>> {
        self.v_parameters_at_u_max.as_ref()
    }

    pub fn u_parameters(&self) -> Option<Vec<T>> {
        self.sorted_parameters(
            self.u_parameters_at_v_min.as_ref(),
            self.u_parameters_at_v_max.as_ref(),
        )
    }

    pub fn v_parameters(&self) -> Option<Vec<T>> {
        self.sorted_parameters(
            self.v_parameters_at_u_min.as_ref(),
            self.v_parameters_at_u_max.as_ref(),
        )
    }

    fn sorted_parameters(&self, min: Option<&Vec<T>>, max: Option<&Vec<T>>) -> Option<Vec<T>> {
        match (min, max) {
            (None, None) => None,
            (None, Some(ma)) => Some(ma.clone()),
            (Some(mi), None) => Some(mi.clone()),
            (Some(mi), Some(ma)) => Some(
                mi.iter()
                    .chain(ma.iter())
                    .sorted_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .dedup()
                    .cloned()
                    .collect(),
            ),
        }
    }
}

/// A struct representing the boundary evaluation of a surface
#[derive(Clone, Debug)]
pub struct BoundaryEvaluation<T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub(crate) u_points_at_v_min: Option<Vec<OPoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) u_points_at_v_max: Option<Vec<OPoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) v_points_at_u_min: Option<Vec<OPoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) v_points_at_u_max: Option<Vec<OPoint<T, DimNameDiff<D, U1>>>>,
}

impl<T: FloatingPoint, D: DimName> BoundaryEvaluation<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Create a new boundary evaluation
    pub fn new(surface: &NurbsSurface<T, D>, constraints: &BoundaryConstraints<T>) -> Self {
        let (u, v) = surface.knots_domain();
        Self {
            u_points_at_v_min: constraints.u_parameters_at_v_min().map(|u_parameters| {
                u_parameters
                    .iter()
                    .map(|u| surface.point_at(*u, v.0))
                    .collect()
            }),
            u_points_at_v_max: constraints.u_parameters_at_v_max().map(|u_parameters| {
                u_parameters
                    .iter()
                    .map(|u| surface.point_at(*u, v.1))
                    .collect()
            }),
            v_points_at_u_min: constraints.v_parameters_at_u_min().map(|v_parameters| {
                v_parameters
                    .iter()
                    .map(|v| surface.point_at(u.0, *v))
                    .collect()
            }),
            v_points_at_u_max: constraints.v_parameters_at_u_max().map(|v_parameters| {
                v_parameters
                    .iter()
                    .map(|v| surface.point_at(u.1, *v))
                    .collect()
            }),
        }
    }
}
