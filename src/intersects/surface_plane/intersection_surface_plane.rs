use nalgebra::{Const, OPoint};

use crate::{intersects::Intersection, prelude::CurveIntersectionSolverOptions};

pub type SurfacePlaneIntersection<T> = Vec<Intersection<OPoint<T, Const<3>>, T, ()>>;

/// Options for surface-plane intersection solver
pub type SurfaceIntersectionSolverOptions<T> = CurveIntersectionSolverOptions<T>;

/*
impl<'a, T> Intersects<'a, &'a Plane<T>> for NurbsSurface<T, Const<4>>
where
    T: FloatingPoint + ArgminFloat + num_traits::Bounded + SubsetOf<f64>,
{
    type Output = anyhow::Result<Vec<NurbsCurve<T, Const<4>>>>;
    type Option = Option<SurfaceIntersectionSolverOptions<T>>;

    /// Find the intersection curves between a surface and a plane
    /// * `plane` - The plane to intersect with
    /// * `options` - Hyperparameters for the intersection solver
    fn find_intersection(&'a self, plane: &'a Plane<T>, _option: Self::Option) -> Self::Output {
        let tess = self.tessellate(Some(AdaptiveTessellationOptions::<T, Const<4>>::default()));
        let its = tess.find_intersection(plane, ())?;

        /*
        let debug = its.polylines.iter().map(|polyline| {
            NurbsCurve3D::polyline(polyline, false)
        }).collect_vec();
        return Ok(debug);
        */

        let projected = its
            .polylines
            .iter()
            .map(|polyline| {
                // use hint to find the closest point
                let mut iter = polyline.iter();
                let first = iter.next().ok_or(anyhow::anyhow!("No first point"))?;
                let uv = self.find_closest_parameter(first, None)?;
                let parameters = iter.try_fold(vec![uv], |mut acc, pt| {
                    let uv = self.find_closest_parameter(pt, Some(*acc.last().unwrap()))?;
                    // let uv = self.find_closest_parameter(pt, None)?;
                    acc.push(uv);
                    anyhow::Ok(acc)
                })?;
                anyhow::Ok(parameters)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let curves = projected
            .iter()
            .map(|parameters| {
                let degree = (parameters.len() - 1).min(2);
                // let degree = (parameters.len() - 1).min(3);
                let parameter_curve = NurbsCurve2D::interpolate(
                    &parameters
                        .iter()
                        .map(|uv| Point2::new(uv.0, uv.1))
                        .collect_vec(),
                    degree,
                )?;

                let pts = parameter_curve.dehomogenized_control_points();
                let pts = pts
                    .iter()
                    .map(|uv| self.point_at(uv.x, uv.y).to_homogeneous().into())
                    .collect_vec();
                NurbsCurve3D::try_new(
                    parameter_curve.degree(),
                    pts,
                    parameter_curve.knots().to_vec(),
                )
            })
            // .map(|polyline| Ok(NurbsCurve3D::polyline(&polyline, false)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        // return Ok(curves);

        let debug = projected
            .iter()
            .map(|params| {
                // println!("polyline: {:?}", params);
                NurbsCurve3D::polyline(
                    &params
                        .iter()
                        .map(|uv| self.point_at(uv.0, uv.1))
                        .collect_vec(),
                    false,
                )
            })
            .collect_vec();
        Ok(curves.into_iter().chain(debug).collect_vec())
    }
}
*/

#[cfg(test)]
mod tests {}
