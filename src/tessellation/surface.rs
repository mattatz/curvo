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
    fn constrained_tessellate(
        &self,
        constraints: Self::Constraint,
        adaptive_options: Self::Option,
    ) -> Self::Output {
        surface_adaptive_tessellate(self, Some(constraints), adaptive_options)
    }
}

/// Tessellate the surface adaptively
fn surface_adaptive_tessellate<T: FloatingPoint, D>(
    s: &NurbsSurface<T, D>,
    constraints: Option<BoundaryConstraints<T>>,
    adaptive_options: Option<AdaptiveTessellationOptions<T>>,
) -> SurfaceTessellation<T, D>
where
    D: DimName,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    let is_adaptive = adaptive_options.is_some();
    let options = adaptive_options.unwrap_or_default();

    // if constraints are provided, we only need to tessellate the surface at the control points
    // otherwise, we need to tessellate the surface at twice the number of control points
    // to ensure that the surface is tessellated enough to capture the curvature
    let min_divs_multiplier = if constraints.is_some() { 1 } else { 2 };

    let us: Vec<_> = if s.u_degree() <= 1 {
        s.u_knots()
            .iter()
            .skip(1)
            .take(s.u_knots().len() - 2)
            .cloned()
            .collect()
    } else {
        let min_u = (s.control_points().len() - 1) * min_divs_multiplier;
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
        let min_v = (s.control_points()[0].len() - 1) * min_divs_multiplier;
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
            // merge_sorted_parameters(us, iu)
            iu
        } else {
            us
        };
        let vs = if let Some(iv) = c.v_parameters() {
            // merge_sorted_parameters(vs, iv)
            iv
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

    let eps = T::from_f64(1e-8).unwrap();

    let pts = vs
        .iter()
        .enumerate()
        .map(|(iv, v)| {
            let u_min = iv == 0;
            let u_max = iv == divs_v;
            let u_constraint = (u_min && u_min_constraint) || (u_max && u_max_constraint);

            let row = us.iter().enumerate().map(|(iu, u)| {
                let v_min = iu == 0;
                let v_max = iu == divs_u;
                let v_constraint = (v_min && v_min_constraint) || (v_max && v_max_constraint);

                let ds = s.rational_derivatives(*u, *v, 1);
                let n = ds[1][0].cross(&ds[0][1]);
                let l = n.norm();
                let norm = if l.is_zero() || !l.is_finite() {
                    let u_p = if v_min {
                        *u + eps
                    } else if v_max {
                        *u - eps
                    } else {
                        *u
                    };
                    let v_p = if u_min {
                        *v + eps
                    } else if u_max {
                        *v - eps
                    } else {
                        *v
                    };
                    s.normal_at(u_p, v_p).normalize()
                } else {
                    n / l
                };

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

#[allow(dead_code)]
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

    // append the remaining elements
    if c0 < p0.len() {
        res.extend_from_slice(&p0[c0..]);
    } else if c1 < p1.len() {
        res.extend_from_slice(&p1[c1..]);
    }

    res
}

#[cfg(test)]
mod tests {
    use std::f64::consts::FRAC_PI_2;

    use nalgebra::{Point3, Rotation3, Translation3, Vector3};

    use crate::{curve::NurbsCurve3D, misc::Transformable, prelude::Interpolation};

    use super::*;

    /// Test that the surface is tessellated correctly with constraints on the boundary
    #[test]
    fn surface_constrained_tessellation() {
        let interpolation_target = vec![
            Point3::new(-1.0, -1.0, 0.),
            Point3::new(1.0, -1.0, 0.),
            Point3::new(1.0, 1.0, 0.),
            Point3::new(-1.0, 1.0, 0.),
            Point3::new(-1.0, 2.0, 0.),
            Point3::new(1.0, 2.5, 0.),
        ];
        let interpolated = NurbsCurve3D::<f64>::interpolate(&interpolation_target, 3).unwrap();

        let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
        let translation = Translation3::new(0., 0., 1.5);
        let m = translation * rotation;
        let front = interpolated.transformed(&(translation.inverse()).into());
        let back = interpolated.transformed(&m.into());

        let (umin, umax) = front.knots_domain();

        let surface = NurbsSurface::try_loft(&[front.clone(), back.clone()], Some(3)).unwrap();

        let u_parameters = (0..=8)
            .map(|i| umin + (umax - umin) * i as f64 / 8.)
            .collect_vec();

        let boundary = BoundaryConstraints::default()
            .with_u_parameters_at_v_min(u_parameters.clone())
            .with_u_parameters_at_v_max(u_parameters.clone());
        let option = AdaptiveTessellationOptions {
            norm_tolerance: 1e-3,
            ..Default::default()
        };
        let tess = surface.constrained_tessellate(boundary, Some(option));
        let vertices = tess.points().clone();

        let front_points = u_parameters
            .iter()
            .map(|u| front.point_at(*u))
            .collect_vec();
        let back_points = u_parameters.iter().map(|u| back.point_at(*u)).collect_vec();

        assert!(front_points.iter().all(|p| vertices.contains(p)));
        assert!(back_points.iter().all(|p| vertices.contains(p)));
    }
}
