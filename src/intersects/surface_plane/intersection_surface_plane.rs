use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{Const, OPoint, Point2};
use simba::scalar::SubsetOf;

use crate::{
    curve::{NurbsCurve2D, NurbsCurve3D},
    intersects::Intersection,
    misc::{FloatingPoint, Plane},
    prelude::{
        AdaptiveTessellationOptions, CurveIntersectionSolverOptions, Interpolation, Intersects,
        NurbsCurve, Tessellation,
    },
    surface::NurbsSurface,
};

pub type SurfacePlaneIntersection<T> = Vec<Intersection<OPoint<T, Const<3>>, T, ()>>;

/// Options for surface-plane intersection solver
pub type SurfaceIntersectionSolverOptions<T> = CurveIntersectionSolverOptions<T>;

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
        let tess = self.tessellate(Some(AdaptiveTessellationOptions {
            norm_tolerance: T::from_f64(1e-2).unwrap(),
            // max_depth: 1,
            ..Default::default()
        }));
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

                /*
                let points = parameters
                    .iter()
                    .map(|uv| self.point_at(uv.0, uv.1))
                    .collect_vec();

                anyhow::Ok(points)
                */
                anyhow::Ok(parameters)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let curves = projected
            .iter()
            .map(|parameters| {
                // let degree = (parameters.len() - 1).min(3);
                let degree = (parameters.len() - 1).min(2);
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
            /*
            .map(|polyline| {
                interpolate_nurbs(
                    &polyline.into_iter().map(|p| p.cast::<f64>()).collect_vec(),
                    3,
                )
                .map(|c| c.cast::<T>())
            })
            */
            // .map(|polyline| Ok(NurbsCurve3D::polyline(&polyline, false)))
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Ok(curves)

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

#[cfg(test)]
mod tests {
    use crate::{misc::Plane, prelude::Intersects, surface::NurbsSurface3D};
    use nalgebra::{Point3, Vector3};

    #[test]
    fn test_plane_surface_intersection() {
        // Create a simple surface that curves upward in the middle
        let control_points = vec![
            vec![
                Point3::new(-1.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(-1.0, 0.0, 0.5).to_homogeneous().into(),
                Point3::new(-1.0, 1.0, 0.0).to_homogeneous().into(),
            ],
            vec![
                Point3::new(0.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(0.0, 0.0, 1.0).to_homogeneous().into(),
                Point3::new(0.0, 1.0, 0.0).to_homogeneous().into(),
            ],
            vec![
                Point3::new(1.0, -1.0, 0.0).to_homogeneous().into(),
                Point3::new(1.0, 0.0, 0.5).to_homogeneous().into(),
                Point3::new(1.0, 1.0, 0.0).to_homogeneous().into(),
            ],
        ];

        let surface = NurbsSurface3D::<f64>::new(
            2,
            2,
            vec![0., 0., 0., 1., 1., 1.],
            vec![0., 0., 0., 1., 1., 1.],
            control_points,
        );

        // Create XY plane at z = 0.3
        let plane = Plane::new(Vector3::z(), 0.3);

        let intersections = surface.find_intersection(&plane, None);
        assert!(intersections.is_ok());

        // For now, let's just check that the function runs without panicking
        // The actual curve extraction logic needs more refinement
        let _curves = intersections.unwrap();
    }
}
