use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    curve::nurbs_curve::refine_knot,
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
        let transposed;
        let (points, knots, degree) = match option.direction {
            UVDirection::U => {
                transposed = self.transposed_control_points();
                (&transposed, self.u_knots(), self.u_degree())
            }
            UVDirection::V => (self.control_points(), self.v_knots(), self.v_degree()),
        };

        let knots_to_insert = vec![option.parameter; degree + 1];

        let n = knots.len() - degree - 2;
        let s = knots.find_knot_span_index(n, degree, option.parameter);

        // Each row is a curve along the direction; its refined control points are those of the
        // head followed by those of the tail. Every row refines the knots alike.
        let mut refined_knots = None;
        let (pts0, pts1): (Vec<_>, Vec<_>) = points
            .iter()
            .map(|row| {
                anyhow::ensure!(row.len() > degree, "Too few control points for curve");
                anyhow::ensure!(
                    knots.len() == row.len() + degree + 1,
                    "Invalid number of knots, got {}, expected {}",
                    knots.len(),
                    row.len() + degree + 1
                );
                let (mut head, knots) = refine_knot(degree, row, knots, &knots_to_insert);
                refined_knots = Some(knots);
                let tail = head.split_off(s + 1);
                Ok((head, tail))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        let mut knots0 = refined_knots.ok_or_else(|| anyhow::anyhow!("No curves"))?;
        let knots1 = knots0[s + 1..].to_vec();
        knots0.truncate(s + degree + 2);

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
