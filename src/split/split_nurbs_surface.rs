use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{
    misc::{transpose_control_points, FloatingPoint},
    surface::{NurbsSurface, UVDirection},
};

use super::{Split, SplitAt};

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

        // Each row is a curve along the direction, and every row refines the knots alike.
        let split = SplitAt::new(knots, degree, option.parameter);
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
                let (row, knots) = split.refine(row, knots);
                refined_knots = Some(knots);
                Ok(split.divide_control_points(row))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .unzip();

        let refined_knots = refined_knots.ok_or_else(|| anyhow::anyhow!("No curves"))?;
        let (knots0, knots1) = split.divide_knots(refined_knots);

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

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::Point4;

    use crate::{
        split::{Split, SplitSurfaceOption},
        surface::{NurbsSurface3D, UVDirection},
    };

    #[test]
    fn the_halves_of_a_split_surface_cover_the_surface() {
        // Rational, of a different degree in u and in v, with repeated interior knots in both.
        let u_knots = vec![0., 0., 0., 0., 1., 2., 2., 2., 3., 4., 4., 4., 4.];
        let v_knots = vec![0., 0., 0., 1., 1., 2., 3., 3., 3.];
        let control_points = (0..9)
            .map(|i| {
                (0..6)
                    .map(|j| {
                        let (x, y) = (i as f64, j as f64);
                        let w = 1. + 0.1 * x + 0.05 * y;
                        Point4::new(x * w, y * w, (x * 0.5).sin() * (y * 0.4).cos() * w, w)
                    })
                    .collect()
            })
            .collect();
        let surface = NurbsSurface3D::new(3, 2, u_knots.clone(), v_knots.clone(), control_points);

        for (direction, knots) in [(UVDirection::U, &u_knots), (UVDirection::V, &v_knots)] {
            let (start, end) = surface.knots_domain_at(direction);
            let mut ts: Vec<f64> = (1..8)
                .map(|i| start + (end - start) * i as f64 / 8.)
                .collect();
            // on the interior knots
            ts.extend(knots.iter().filter(|k| start < **k && **k < end));
            for t in ts {
                let (head, tail) = surface
                    .try_split(SplitSurfaceOption::new(t, direction))
                    .unwrap();
                assert_eq!(head.knots_domain_at(direction), (start, t));
                assert_eq!(tail.knots_domain_at(direction), (t, end));
                for half in [head, tail] {
                    assert_eq!((half.u_degree(), half.v_degree()), (3, 2));
                    // the other direction is left as it was
                    let other = direction.opposite();
                    assert_eq!(half.knots_domain_at(other), surface.knots_domain_at(other));
                    let ((u0, u1), (v0, v1)) = half.knots_domain();
                    for i in 0..=6 {
                        for j in 0..=6 {
                            let u = u0 + (u1 - u0) * i as f64 / 6.;
                            let v = v0 + (v1 - v0) * j as f64 / 6.;
                            assert_relative_eq!(
                                half.point_at(u, v),
                                surface.point_at(u, v),
                                epsilon = 1e-9
                            );
                        }
                    }
                }
            }
        }
    }
}
