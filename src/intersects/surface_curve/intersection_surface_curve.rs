use argmin::core::ArgminFloat;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Vector3, U1,
};

use crate::{
    curve::NurbsCurve,
    intersects::solve::{closest_to_an_end, curve_scale, surface_scale, Scales},
    misc::FloatingPoint,
    prelude::{
        BoundingBoxTraversal, CurveBoundingBoxTree, CurveIntersectionSolverOptions, Intersects,
        SurfaceBoundingBoxTree, SurfaceCurveIntersection,
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

        let ta = SurfaceBoundingBoxTree::with_divisions(
            self,
            UVDirection::U,
            options.knot_domain_division,
        );
        let tb = CurveBoundingBoxTree::with_divisions(other, options.knot_domain_division);

        let (size, [u, v]) = surface_scale(self);
        let (curve_size, t) = curve_scale(other);
        // the parameters are those of the curve, then those of the surface
        let scales = Scales::new(&[size, curve_size], [t, u, v], options.minimum_distance);

        // a curve closer to the surface than the minimum distance intersects it, even if their
        // bounding boxes do not overlap
        let traversed =
            BoundingBoxTraversal::try_traverse_with_tolerance(ta, tb, scales.minimum_distance())?;

        let candidates = traversed.into_pairs_iter().flat_map(|(a, b)| {
            let (surface, curve) = (a.surface_owned(), b.curve_owned());
            let problem = SurfaceCurveIntersectionProblem::new(&surface, &curve);
            let (u, v) = surface.knots_domain();
            let found = scales.solve_leaf(&problem, &options, [curve.knots_domain(), u, v]);
            found.into_iter().flatten().map(|(param, past)| {
                let (t, (u, v)) = closest_to_an_end(
                    (param.x, past[0]),
                    ((param.y, param.z), past[1] || past[2]),
                    |(u, v)| other.find_closest_parameter(&self.point_at(*u, *v)).ok(),
                    |t| {
                        let hint = Some((param.y, param.z));
                        self.find_closest_parameter(&other.point_at(*t), hint).ok()
                    },
                );
                Vector3::new(t, u, v)
            })
        });

        let distance_at =
            |param: &Vector3<T>| (self.point_at(param.y, param.z) - other.point_at(param.x)).norm();
        let intersections = scales
            .intersections(candidates, 0, distance_at)
            .into_iter()
            .map(|param| {
                let (t, uv) = (param.x, (param.y, param.z));
                SurfaceCurveIntersection::new(
                    (self.point_at(uv.0, uv.1), uv),
                    (other.point_at(t), t),
                )
            })
            .collect();
        Ok(intersections)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Point3, Vector3};

    use crate::{curve::NurbsCurve3D, prelude::HasIntersection, surface::NurbsSurface3D};

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
        let intersections = sphere.find_intersection(&tangent, None).unwrap();
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].a().0 - Point3::new(0., 0.6, 0.8)).norm() < 1e-2);
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
            let intersections = sphere.find_intersection(&line, None).unwrap();
            assert_eq!(intersections.len(), 2, "scaled by {scale}");
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
        let intersections = sphere.find_intersection(&axis, None).unwrap();
        assert_eq!(intersections.len(), 2);
        assert!((intersections[0].a().0 - Point3::new(-1., 0., 0.)).norm() < 1e-5);
        assert!((intersections[1].a().0 - Point3::new(1., 0., 0.)).norm() < 1e-5);
    }
}
