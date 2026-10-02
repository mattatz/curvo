use argmin::core::ArgminFloat;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Vector3, U1,
};

use crate::{
    curve::NurbsCurve,
    intersects::solve::{closest_in_groups, curve_scale, surface_scale, Scales},
    misc::FloatingPoint,
    prelude::{
        BoundingBoxTraversal, CurveBoundingBoxTree, CurveIntersectionSolverOptions,
        HasIntersection, Intersects, SurfaceBoundingBoxTree, SurfaceCurveIntersection,
    },
    surface::{NurbsSurface, UVDirection},
};

use super::SurfaceCurveIntersectionProblem;

impl<'a, T, D> Intersects<'a, &'a NurbsCurve<T, D>> for NurbsSurface<T, D>
where
    T: FloatingPoint + ArgminFloat,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = anyhow::Result<Vec<SurfaceCurveIntersection<OPoint<T, DimNameDiff<D, U1>>, T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    /// Find the intersections between the surface and the curve.
    /// CAUTION: This method is experimental and may not work as expected.
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    /// use approx::assert_relative_eq;
    /// let unit_sphere = NurbsSurface3D::try_sphere(
    ///     &Point3::origin(),
    ///     &Vector3::x(),
    ///     &Vector3::y(),
    ///     1.
    /// ).unwrap();
    /// let line = NurbsCurve3D::polyline(&[
    ///     Point3::new(-2.0, 0.0, 0.0),
    ///     Point3::new(2.0, 0.0, 0.0),
    /// ], false);
    /// let intersections = unit_sphere.find_intersection(&line, None).unwrap();
    /// assert_eq!(intersections.len(), 2);
    ///
    /// let it0 = &intersections[0];
    /// let p0 = it0.a().0;
    /// assert_relative_eq!(p0.x, -1., epsilon = 1e-5);
    ///
    /// let it1 = &intersections[1];
    /// let p1 = it1.a().0;
    /// assert_relative_eq!(p1.x, 1., epsilon = 1e-5);
    /// ```
    fn find_intersection(
        &'a self,
        other: &'a NurbsCurve<T, D>,
        option: Self::Option,
    ) -> Self::Output {
        let options = option.unwrap_or_default();

        let div = T::one() / T::from_usize(options.knot_domain_division).unwrap();
        let interval = self.knots_domain_interval();
        let ta = SurfaceBoundingBoxTree::new(
            self,
            UVDirection::U,
            Some((interval.0 * div, interval.1 * div)),
        );
        let tb = CurveBoundingBoxTree::new(other, Some(other.knots_domain_interval() * div));

        let traversed = BoundingBoxTraversal::try_traverse(ta, tb)?;

        let (size, [u, v]) = surface_scale(self);
        let (curve_size, t) = curve_scale(other);
        // the parameters are those of the curve, then those of the surface
        let scales = Scales::new(&[size, curve_size], [t, u, v]);

        let candidates = traversed.into_pairs_iter().filter_map(|(a, b)| {
            let surface = a.surface_owned();
            let curve = b.curve_owned();

            let div = T::from_f64(0.5).unwrap();

            let d = curve.knots_domain();
            let curve_parameter = (d.0 + d.1) * div;

            let (u, v) = surface.knots_domain();
            let surface_parameter = ((u.0 + u.1) * div, (v.0 + v.1) * div);

            let problem = SurfaceCurveIntersectionProblem::new(&surface, &curve);

            // Define initial parameter vector
            let init_param =
                Vector3::new(curve_parameter, surface_parameter.0, surface_parameter.1);

            // Run solver
            let param = scales.solve(problem, &options, init_param)?;

            // An intersection at the end of a domain, a pole of a sphere for one, is found a
            // hair inside it or a hair outside, so the parameters are clamped rather than
            // refused: how far apart the points there are decides. What was found past the end
            // of the curve, or the edge of the surface, is where the two would meet if that one
            // went on, so the candidate is that end or edge and the closest point of the other.
            let t = other.knots().clamp(other.degree(), param.x);
            let u = self.u_knots().clamp(self.u_degree(), param.y);
            let v = self.v_knots().clamp(self.v_degree(), param.z);
            let (t, (u, v)) = match (t != param.x, u != param.y || v != param.z) {
                (true, false) => {
                    let closest = self.find_closest_parameter(&other.point_at(t), Some((u, v)));
                    (t, closest.unwrap_or((u, v)))
                }
                (false, true) => {
                    let closest = other.find_closest_parameter(&self.point_at(u, v));
                    (closest.unwrap_or(t), (u, v))
                }
                _ => (t, (u, v)),
            };
            let p0 = self.point_at(u, v);
            let p1 = other.point_at(t);
            Some(SurfaceCurveIntersection::new((p0, (u, v)), (p1, t)))
        });

        // Those whose points are closer than the minimum distance, relative to the size of the
        // geometry, and of those that are one intersection the closest. Two candidates are one
        // intersection when the curve and the surface are still in contact halfway between
        // them. That is so of an intersection found from several pairs of leaves, and of the
        // candidates found all along the stretch where the two touch and stay closer than the
        // minimum distance.
        let minimum_distance = scales.distance(options.minimum_distance);
        Ok(closest_in_groups(
            candidates,
            |it| (&it.a().0 - &it.b().0).norm(),
            minimum_distance,
            |it| it.b().1,
            scales.is_closed(0),
            |x, y, across| {
                let ((xu, xv), xt) = (x.a().1, x.b().1);
                let ((yu, yv), yt) = (y.a().1, y.b().1);
                let t = if across {
                    scales.halfway_across(0, xt, yt)
                } else {
                    (xt + yt) * T::from_f64(0.5).unwrap()
                };
                let on_surface =
                    self.point_at(scales.halfway(1, xu, yu), scales.halfway(2, xv, yv));
                let on_curve = other.point_at(t);
                (on_surface - on_curve).norm() < minimum_distance
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use crate::{curve::NurbsCurve3D, surface::NurbsSurface3D};

    use super::*;

    #[test]
    fn a_line_that_touches_a_sphere_intersects_it_once() {
        let sphere =
            NurbsSurface3D::<f64>::try_sphere(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.)
                .unwrap();
        let tangent = NurbsCurve3D::polyline(
            &[Point3::new(-2., 0.6, 0.8), Point3::new(2., 0.6, 0.8)],
            false,
        );
        for _ in 0..10 {
            let intersections = sphere.find_intersection(&tangent, None).unwrap();
            assert_eq!(intersections.len(), 1);
            assert!((intersections[0].a().0 - Point3::new(0., 0.6, 0.8)).norm() < 1e-2);
        }
    }

    #[test]
    fn intersections_do_not_depend_on_the_scale() {
        for scale in [1e-3, 1e3] {
            let sphere = NurbsSurface3D::<f64>::try_sphere(
                &Point3::origin(),
                &Vector3::x(),
                &Vector3::y(),
                scale,
            )
            .unwrap();
            let line = NurbsCurve3D::polyline(
                &[
                    Point3::new(-2., 0.3, 0.2) * scale,
                    Point3::new(2., -0.4, 0.5) * scale,
                ],
                false,
            );
            for _ in 0..5 {
                let intersections = sphere.find_intersection(&line, None).unwrap();
                assert_eq!(intersections.len(), 2, "scaled by {scale}");
            }
        }
    }

    #[test]
    fn intersections_at_the_poles_of_a_sphere_are_found() {
        // The poles are at the ends of the domain of the sphere, and every patch around a pole
        // finds the intersection there.
        let sphere =
            NurbsSurface3D::<f64>::try_sphere(&Point3::origin(), &Vector3::x(), &Vector3::y(), 1.)
                .unwrap();
        let axis =
            NurbsCurve3D::polyline(&[Point3::new(-2., 0., 0.), Point3::new(2., 0., 0.)], false);
        // The bounding box trees are divided at random, so this is run again and again.
        for _ in 0..10 {
            let intersections = sphere.find_intersection(&axis, None).unwrap();
            assert_eq!(intersections.len(), 2);
            assert!((intersections[0].a().0 - Point3::new(-1., 0., 0.)).norm() < 1e-5);
            assert!((intersections[1].a().0 - Point3::new(1., 0., 0.)).norm() < 1e-5);
        }
    }
}
