use super::adaptive_tessellation_option::AdaptiveTessellationOptions;
use super::adaptive_tessellation_processor::AdaptiveTessellationProcessor;
use super::surface_tessellation::SurfaceTessellation;
use super::{adaptive_tessellation_node::AdaptiveTessellationNode, Tessellation};
use super::{ConstrainedTessellation, SurfacePoint};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Vector2, U1,
};

use crate::{misc::FloatingPoint, surface::NurbsSurface};

impl<T: FloatingPoint, D: DimName> Tessellation for NurbsSurface<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Option = Option<AdaptiveTessellationOptions<T>>;
    type Output = SurfaceTessellation<T, D>;

    /// Tessellate the surface into a meshable form
    /// if adaptive_options is None, the surface will be tessellated at control points
    /// or else it will be tessellated adaptively based on the options
    /// this `adaptive` means that the surface will be tessellated based on the curvature of the surface
    fn tessellate(&self, adaptive_options: Self::Option) -> Self::Output {
        let nodes = surface_adaptive_tessellate(self, None, adaptive_options);
        SurfaceTessellation::new(self, &nodes)
    }
}

/// A struct representing constraints at the seam of a surface tessellation
#[derive(Clone, Debug)]
pub struct SeamConstraints<T: FloatingPoint> {
    v_parameters_at_u_min: Option<Vec<T>>,
    u_parameters_at_v_min: Option<Vec<T>>,
    v_parameters_at_u_max: Option<Vec<T>>,
    u_parameters_at_v_max: Option<Vec<T>>,
}

impl<T: FloatingPoint> SeamConstraints<T> {
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

impl<T: FloatingPoint, D: DimName> ConstrainedTessellation for NurbsSurface<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Constraint = SeamConstraints<T>;
    type Option = Option<AdaptiveTessellationOptions<T>>;
    type Output = SurfaceTessellation<T, D>;

    /// Tessellate the surface into a meshable form with constraints on the seam
    /// Parameters on the seam are computed before surface tessellation to ensure that the subdivided vertices are the same at the seam
    fn constrained_tessalate(
        &self,
        constraints: Self::Constraint,
        adaptive_options: Self::Option,
    ) -> Self::Output {
        let nodes = surface_adaptive_tessellate(self, Some(constraints), adaptive_options);
        SurfaceTessellation::new(self, &nodes)
    }
}

/// Tessellate the surface adaptively
fn surface_adaptive_tessellate<T: FloatingPoint, D: DimName>(
    s: &NurbsSurface<T, D>,
    constraints: Option<SeamConstraints<T>>,
    adaptive_options: Option<AdaptiveTessellationOptions<T>>,
) -> Vec<AdaptiveTessellationNode<T, D>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let is_adaptive = adaptive_options.is_some();
    let options = adaptive_options.unwrap_or_default();

    let us: Vec<_> = if s.u_degree() <= 1 {
        s.u_knots()
            .iter()
            .skip(1)
            .take(s.u_knots().len() - 2)
            .cloned()
            .collect()
    } else {
        let min_u = (s.control_points().len() - 1) * 2;
        let divs_u = options.min_divs_u.max(min_u);
        let (umin, umax) = s.u_knots_domain();
        let du = (umax - umin) / T::from_usize(divs_u).unwrap();
        (0..=divs_u)
            .map(|i| umin + du * T::from_usize(i).unwrap())
            .collect()
    };

    let vs: Vec<_> = if s.v_degree() <= 1 {
        s.v_knots()
            .iter()
            .skip(1)
            .take(s.v_knots().len() - 2)
            .cloned()
            .collect()
    } else {
        let min_v = (s.control_points()[0].len() - 1) * 2;
        let divs_v = options.min_divs_v.max(min_v);
        let (vmin, vmax) = s.v_knots_domain();
        let dv = (vmax - vmin) / T::from_usize(divs_v).unwrap();
        (0..=divs_v)
            .map(|i| vmin + dv * T::from_usize(i).unwrap())
            .collect()
    };

    // insert seam parameters to us & vs if constraints are provided
    let (us, vs) = if let Some(c) = constraints {
        let us = if let Some(iu) = c.u_parameters() {
            us
        } else {
            us
        };
        let vs = if let Some(iv) = c.v_parameters() {
            vs
        } else {
            vs
        };
        (us, vs)
    } else {
        (us, vs)
    };

    let pts = vs
        .iter()
        .map(|v| {
            let row = us.iter().map(|u| {
                let ds = s.rational_derivatives(*u, *v, 1);
                let norm = ds[1][0].cross(&ds[0][1]).normalize();
                SurfacePoint::new(Vector2::new(*u, *v), ds[0][0].clone().into(), norm, false)
            });
            row.collect_vec()
        })
        .collect_vec();

    let divs_u = us.len() - 1;
    let divs_v = vs.len() - 1;
    let pts = &pts;

    let nodes = (0..divs_v)
        .flat_map(|i| {
            let iv = divs_v - i;
            (0..divs_u).map(move |iu| {
                let corners = [
                    pts[iv - 1][iu].clone(),
                    pts[iv - 1][iu + 1].clone(),
                    pts[iv][iu + 1].clone(),
                    pts[iv][iu].clone(),
                ];
                let index = i * divs_u + iu;
                AdaptiveTessellationNode::new(index, corners, None)
            })
        })
        .collect_vec();

    if !is_adaptive {
        nodes
    } else {
        let mut processor = AdaptiveTessellationProcessor::new(s, nodes);

        for iv in 0..divs_v {
            for iu in 0..divs_u {
                let index = iv * divs_u + iu;
                let s = processor.south(index, iv, divs_u, divs_v).map(|n| n.id());
                let e = processor.east(index, iu, divs_u).map(|n| n.id());
                let n = processor.north(index, iv, divs_u).map(|n| n.id());
                let w = processor.west(index, iv).map(|n| n.id());
                let node = processor.nodes_mut().get_mut(index).unwrap();
                node.neighbors = [s, e, n, w];
                processor.divide(index, &options);
            }
        }

        processor.into_nodes()
    }
}
