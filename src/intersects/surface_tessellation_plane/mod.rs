use nalgebra::Point3;

use crate::{
    misc::{FloatingPoint, Plane},
    prelude::{Intersects, SurfaceTessellation3D},
};

impl<'a, T: FloatingPoint> Intersects<'a, &'a Plane<T>> for SurfaceTessellation3D<T> {
    type Output = anyhow::Result<Vec<Point3<T>>>;
    type Option = ();

    /// Find the intersection points between a surface tessellation and a plane
    fn find_intersection(&'a self, other: &'a Plane<T>, option: Self::Option) -> Self::Output {
        todo!()
    }
}
