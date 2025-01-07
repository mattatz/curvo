use super::adaptive_tessellation_option::AdaptiveTessellationOptions;
use super::adaptive_tessellation_processor::AdaptiveTessellationProcessor;
use super::boundary_constraints::BoundaryConstraints;
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
        surface_adaptive_tessellate(self, None, adaptive_options)
    }
}

impl<T: FloatingPoint, D: DimName> ConstrainedTessellation for NurbsSurface<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Constraint = BoundaryConstraints<T>;
    type Option = Option<AdaptiveTessellationOptions<T>>;
    type Output = SurfaceTessellation<T, D>;

    /// Tessellate the surface into a meshable form with constraints on the boundary
    /// Parameters on the boundary are computed before surface tessellation to ensure that the subdivided vertices are the same at the boundary
    fn constrained_tessalate(
        &self,
        constraints: Self::Constraint,
        adaptive_options: Self::Option,
    ) -> Self::Output {
        surface_adaptive_tessellate(self, Some(constraints), adaptive_options)
    }
}

/// Tessellate the surface adaptively
fn surface_adaptive_tessellate<T: FloatingPoint, D: DimName>(
    s: &NurbsSurface<T, D>,
    constraints: Option<BoundaryConstraints<T>>,
    adaptive_options: Option<AdaptiveTessellationOptions<T>>,
) -> SurfaceTessellation<T, D>
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

    // insert boundary parameters to us & vs if constraints are provided
    let (us, vs) = if let Some(c) = constraints.as_ref() {
        let us = if let Some(iu) = c.u_parameters() {
            merge_sorted_parameters(us, iu)
        } else {
            us
        };
        let vs = if let Some(iv) = c.v_parameters() {
            merge_sorted_parameters(vs, iv)
        } else {
            vs
        };
        (us, vs)
    } else {
        (us, vs)
    };

    let (u_min_constraint, u_max_constraint, v_min_constraint, v_max_constraint) = constraints
        .as_ref()
        .map(|c| {
            let u_in_v_min = c.u_parameters_at_v_min().is_some();
            let u_in_v_max = c.u_parameters_at_v_max().is_some();
            let v_in_u_min = c.v_parameters_at_u_min().is_some();
            let v_in_u_max = c.v_parameters_at_u_max().is_some();
            (u_in_v_min, u_in_v_max, v_in_u_min, v_in_u_max)
        })
        .unwrap_or((false, false, false, false));

    let divs_u = us.len() - 1;
    let divs_v = vs.len() - 1;

    let pts = vs
        .iter()
        .enumerate()
        .map(|(iv, v)| {
            let u_min = iv == 0;
            let u_max = iv == divs_v - 1;
            let u_constraint = (u_min && u_min_constraint) || (u_max && u_max_constraint);

            let row = us.iter().enumerate().map(|(iu, u)| {
                let ds = s.rational_derivatives(*u, *v, 1);
                let norm = ds[1][0].cross(&ds[0][1]).normalize();
                let v_min = iu == 0;
                let v_max = iu == divs_u - 1;
                let v_constraint = (v_min && v_min_constraint) || (v_max && v_max_constraint);
                SurfacePoint::new(Vector2::new(*u, *v), ds[0][0].clone().into(), norm, false)
                    .with_constraints(u_constraint, v_constraint)
                    .with_boundary(u_min, u_max, v_min, v_max)
            });
            row.collect_vec()
        })
        .collect_vec();

    let pts = &pts;

    let nodes = (0..divs_v)
        .flat_map(|iv: usize| {
            let iv_r = divs_v - iv;
            (0..divs_u).map(move |iu| {
                let corners = [
                    pts[iv_r - 1][iu].clone(),
                    pts[iv_r - 1][iu + 1].clone(),
                    pts[iv_r][iu + 1].clone(),
                    pts[iv_r][iu].clone(),
                ];
                let index = iv * divs_u + iu;
                let s = south(index, iv, divs_u, divs_v);
                let e = east(index, iu, divs_u);
                let n = north(index, iv, divs_u);
                let w = west(index, iv);
                AdaptiveTessellationNode::new(index, corners, [s, e, n, w])
            })
        })
        .collect_vec();

    let nodes = if !is_adaptive {
        nodes
    } else {
        let mut processor = AdaptiveTessellationProcessor::new(s, nodes);

        for iv in 0..divs_v {
            for iu in 0..divs_u {
                let index = iv * divs_u + iu;
                processor.divide(index, &options);
            }
        }

        processor.into_nodes()
    };

    SurfaceTessellation::new(s, &nodes, constraints)
}

fn north(index: usize, iv: usize, divs_u: usize) -> Option<usize> {
    if iv == 0 {
        None
    } else {
        Some(index - divs_u)
    }
}

fn south(index: usize, iv: usize, divs_u: usize, divs_v: usize) -> Option<usize> {
    if iv == divs_v - 1 {
        None
    } else {
        Some(index + divs_u)
    }
}

fn east(index: usize, iu: usize, divs_u: usize) -> Option<usize> {
    if iu == divs_u - 1 {
        None
    } else {
        Some(index + 1)
    }
}

fn west(index: usize, iv: usize) -> Option<usize> {
    if iv == 0 {
        None
    } else {
        Some(index - 1)
    }
}

/// Merge two sorted vectors
fn merge_sorted_parameters<T: FloatingPoint>(p0: Vec<T>, p1: Vec<T>) -> Vec<T> {
    let mut c0 = 0;
    let mut c1 = 0;
    let mut res = vec![];
    while c0 < p0.len() && c1 < p1.len() {
        if p0[c0] < p1[c1] {
            res.push(p0[c0]);
            c0 += 1;
        } else if p0[c0] > p1[c1] {
            res.push(p1[c1]);
            c1 += 1;
        } else {
            res.push(p0[c0]);
            c0 += 1;
            c1 += 1;
        }
    }
    res
}
