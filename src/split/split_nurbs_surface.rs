use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    curve::NurbsCurve,
    misc::{transpose_control_points, FloatingPoint},
    surface::{NurbsSurface, UVDirection},
};

use super::Split;

/// Option for splitting a surface
#[derive(Clone, Debug)]
pub struct SplitSurfaceOption<T: FloatingPoint> {
    // parameter to split
    pub parameter: T,
    // split direction
    pub direction: UVDirection,
}

impl<T: FloatingPoint> SplitSurfaceOption<T> {
    pub fn new(parameter: T, direction: UVDirection) -> Self {
        Self {
            parameter,
            direction,
        }
    }
}

impl<T: FloatingPoint, D: DimName> Split for NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Option = SplitSurfaceOption<T>;

    /// Split the surface into two surfaces before and after the parameter
    fn try_split(&self, option: Self::Option) -> anyhow::Result<(Self, Self)> {
        let (points, knots, degree) = match option.direction {
            UVDirection::U => {
                let transposed = self.transposed_control_points();
                (transposed, self.u_knots(), self.u_degree())
            }
            UVDirection::V => {
                let pts = self.control_points();
                (pts.clone(), self.v_knots(), self.v_degree())
            }
        };

        let knots_to_insert = (0..=degree).map(|_| option.parameter).collect_vec();

        let n = knots.len() - degree - 2;
        let s = knots.find_knot_span_index(n, degree, option.parameter);

        let curves = points
            .iter()
            .map(|row| {
                let mut curve = NurbsCurve::try_new(degree, row.clone(), knots.to_vec())?;
                curve.try_refine_knot(knots_to_insert.clone())?;
                Ok(curve)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let (pts0, pts1): (Vec<_>, Vec<_>) = curves
            .iter()
            .map(|curve| {
                let p0 = curve.control_points()[0..=s].to_vec();
                let p1 = curve.control_points()[s + 1..].to_vec();
                (p0, p1)
            })
            .unzip();

        let last = curves.last().ok_or_else(|| anyhow::anyhow!("No curves"))?;
        let knots0 = last.knots().as_slice()[0..=(s + degree + 1)].to_vec();
        let knots1 = last.knots().as_slice()[s + 1..].to_vec();

        match option.direction {
            UVDirection::U => Ok((
                Self::new(
                    degree,
                    self.v_degree(),
                    knots0,
                    self.v_knots().to_vec(),
                    transpose_control_points(&pts0),
                ),
                Self::new(
                    degree,
                    self.v_degree(),
                    knots1,
                    self.v_knots().to_vec(),
                    transpose_control_points(&pts1),
                ),
            )),
            UVDirection::V => Ok((
                Self::new(
                    self.u_degree(),
                    degree,
                    self.u_knots().to_vec(),
                    knots0,
                    pts0,
                ),
                Self::new(
                    self.u_degree(),
                    degree,
                    self.u_knots().to_vec(),
                    knots1,
                    pts1,
                ),
            )),
        }
    }
}
