use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Vector2, U1,
};

use crate::{misc::FloatingPoint, surface::NurbsSurface};

use super::SurfacePoint;

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
    pub(crate) u_points_at_v_min: Option<Vec<SurfacePoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) u_points_at_v_max: Option<Vec<SurfacePoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) v_points_at_u_min: Option<Vec<SurfacePoint<T, DimNameDiff<D, U1>>>>,
    pub(crate) v_points_at_u_max: Option<Vec<SurfacePoint<T, DimNameDiff<D, U1>>>>,
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

        let evaluate = |uv: Vector2<T>| {
            let deriv = surface.rational_derivatives(uv.x, uv.y, 1);
            let pt = deriv[0][0].clone();
            let norm = deriv[1][0].cross(&deriv[0][1]);
            let is_normal_degenerated = norm.magnitude_squared() < T::default_epsilon();
            SurfacePoint::new(uv, pt.into(), norm, is_normal_degenerated)
        };

        Self {
            u_points_at_v_min: constraints.u_parameters_at_v_min().map(|u_parameters| {
                u_parameters
                    .iter()
                    .map(|u| evaluate(Vector2::new(*u, v.0)))
                    .collect()
            }),
            u_points_at_v_max: constraints.u_parameters_at_v_max().map(|u_parameters| {
                u_parameters
                    .iter()
                    .map(|u| evaluate(Vector2::new(*u, v.1)))
                    .collect()
            }),
            v_points_at_u_min: constraints.v_parameters_at_u_min().map(|v_parameters| {
                v_parameters
                    .iter()
                    .map(|v| evaluate(Vector2::new(u.0, *v)))
                    .collect()
            }),
            v_points_at_u_max: constraints.v_parameters_at_u_max().map(|v_parameters| {
                v_parameters
                    .iter()
                    .map(|v| evaluate(Vector2::new(u.1, *v)))
                    .collect()
            }),
        }
    }

    pub fn closest_point_at_u_min(
        &self,
        point: &SurfacePoint<T, DimNameDiff<D, U1>>,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        if let Some(u_points) = self.u_points_at_v_min.as_ref() {
            self.closest_point(point, u_points)
        } else {
            point.clone()
        }
    }

    pub fn closest_point_at_u_max(
        &self,
        point: &SurfacePoint<T, DimNameDiff<D, U1>>,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        if let Some(u_points) = self.u_points_at_v_max.as_ref() {
            self.closest_point(point, u_points)
        } else {
            point.clone()
        }
    }

    pub fn closest_point_at_v_min(
        &self,
        point: &SurfacePoint<T, DimNameDiff<D, U1>>,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        if let Some(v_points) = self.v_points_at_u_min.as_ref() {
            self.closest_point(point, v_points)
        } else {
            point.clone()
        }
    }

    pub fn closest_point_at_v_max(
        &self,
        point: &SurfacePoint<T, DimNameDiff<D, U1>>,
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        if let Some(v_points) = self.v_points_at_u_max.as_ref() {
            self.closest_point(point, v_points)
        } else {
            point.clone()
        }
    }

    /// Find the closest point to the given point from the list of points
    fn closest_point(
        &self,
        point: &SurfacePoint<T, DimNameDiff<D, U1>>,
        points: &[SurfacePoint<T, DimNameDiff<D, U1>>],
    ) -> SurfacePoint<T, DimNameDiff<D, U1>> {
        points
            .iter()
            .map(|p| {
                let dist = (p.point() - point.point()).norm_squared();
                (p, dist)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0
            .clone()
    }
}
