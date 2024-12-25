use super::adaptive_tessellation_option::AdaptiveTessellationOptions;
use super::adaptive_tessellation_processor::AdaptiveTessellationProcessor;
use super::surface_tessellation::SurfaceTessellation;
use super::SurfacePoint;
use super::{adaptive_tessellation_node::AdaptiveTessellationNode, Tessellation};
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
        let is_adaptive = adaptive_options.is_some();
        let options = adaptive_options.unwrap_or_default();

        let us: Vec<_> = if self.u_degree() <= 1 {
            self.u_knots()
                .iter()
                .skip(1)
                .take(self.u_knots().len() - 2)
                .cloned()
                .collect()
        } else {
            let min_u = (self.control_points().len() - 1) * 2;
            let divs_u = options.min_divs_u.max(min_u);
            let (umin, umax) = self.u_knots_domain();
            let du = (umax - umin) / T::from_usize(divs_u).unwrap();
            (0..=divs_u)
                .map(|i| umin + du * T::from_usize(i).unwrap())
                .collect()
        };

        let vs: Vec<_> = if self.v_degree() <= 1 {
            self.v_knots()
                .iter()
                .skip(1)
                .take(self.v_knots().len() - 2)
                .cloned()
                .collect()
        } else {
            let min_v = (self.control_points()[0].len() - 1) * 2;
            let divs_v = options.min_divs_v.max(min_v);
            let (vmin, vmax) = self.v_knots_domain();
            let dv = (vmax - vmin) / T::from_usize(divs_v).unwrap();
            (0..=divs_v)
                .map(|i| vmin + dv * T::from_usize(i).unwrap())
                .collect()
        };

        let pts = vs
            .iter()
            .map(|v| {
                let row = us.iter().map(|u| {
                    let ds = self.rational_derivatives(*u, *v, 1);
                    let norm = ds[1][0].cross(&ds[0][1]).normalize();
                    SurfacePoint {
                        point: ds[0][0].clone().into(),
                        normal: norm,
                        uv: Vector2::new(*u, *v),
                        is_normal_degenerated: false,
                    }
                });
                row.collect_vec()
            })
            .collect_vec();

        let divs_u = us.len() - 1;
        let divs_v = vs.len() - 1;
        let pts = &pts;

        let divs = (0..divs_v)
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

        let nodes = if !is_adaptive {
            divs
        } else {
            let mut processor = AdaptiveTessellationProcessor {
                surface: self,
                nodes: divs,
            };

            for i in 0..divs_v {
                for j in 0..divs_u {
                    let ci = i * divs_u + j;
                    let s = processor.south(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let e = processor.east(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let n = processor.north(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let w = processor.west(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let node = processor.nodes.get_mut(ci).unwrap();
                    node.neighbors = [s, e, n, w];
                    processor.divide(ci, &options);
                }
            }

            processor.nodes
        };

        SurfaceTessellation::new(self, &nodes)
    }
}
