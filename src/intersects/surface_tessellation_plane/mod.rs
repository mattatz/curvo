use nalgebra::U3;
use simba::scalar::SubsetOf;

use crate::prelude::{MeshPlaneIntersection, PolygonMesh};
use crate::{
    misc::{FloatingPoint, Plane},
    prelude::{Intersects, SurfaceTessellation3D},
};

impl<'a, T: FloatingPoint + SubsetOf<f64>> Intersects<'a, &'a Plane<T>>
    for SurfaceTessellation3D<T>
{
    type Output = anyhow::Result<MeshPlaneIntersection<T>>;
    type Option = ();

    /// Find the intersection points between a surface tessellation and a plane
    fn find_intersection(&'a self, other: &'a Plane<T>, _option: Self::Option) -> Self::Output {
        let mesh: PolygonMesh<T, U3> = self.clone().into();
        let it = mesh.find_intersection(other, ());
        it
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use nalgebra::{Point4, Vector3};
    use rand::Rng;

    use crate::{
        knot::KnotVector,
        prelude::{AdaptiveTessellationOptions, Tessellation},
        surface::NurbsSurface3D,
    };

    use super::*;

    /// Test the intersection between a surface tessellation and a plane
    #[test]
    fn test_surface_tessellation_plane() {
        let degree = 3;
        let n: usize = 6;
        let goal = n + degree + 1;
        let knots = KnotVector::uniform(goal - degree * 2, degree);
        let hn = (n - 1) as f64 / 2.;
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
        let pts = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let x = i as f64 - hn;
                        let y = (rng.random::<f64>() - 0.5) * 2.;
                        let z = (j as f64) - hn;
                        Point4::new(x, y, z, 1.)
                    })
                    .collect_vec()
            })
            .collect_vec();
        let surface = NurbsSurface3D::new(degree, degree, knots.to_vec(), knots.to_vec(), pts);
        let plane = Plane::new(Vector3::y(), 0.0);
        let tess = surface.tessellate(Some(
            AdaptiveTessellationOptions::<_>::default().with_norm_tolerance(1e-3),
        ));
        let its = tess.find_intersection(&plane, ());
        assert!(its.is_ok());
        let its = its.unwrap();
        assert_eq!(its.polylines.len(), 385);
    }
}
