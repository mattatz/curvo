use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, Point2, U1,
};
use spade::{
    handles::FixedVertexHandle, ConstrainedDelaunayTriangulation, DelaunayTriangulation,
    Point2 as SP2, SpadeNum, Triangulation,
};

use crate::{misc::FloatingPoint, region::Region};

use super::Tessellation;

type Tri<T> = ConstrainedDelaunayTriangulation<SP2<T>>;

impl<T: FloatingPoint + SpadeNum> Tessellation for Region<T> {
    type Option = Option<T>;
    type Output = ();

    fn tessellate(&self, tolerance: Option<T>) -> Self::Output {
        let mut t = Tri::default();

        [
            &[self.exterior()],
            self.interiors().iter().collect_vec().as_slice(),
        ]
        .concat()
        .into_iter()
        .for_each(|curve| {
            let handles = curve
                .tessellate(tolerance)
                .iter()
                .enumerate()
                .map(|(i, pt)| {
                    t.insert_with_hint(SP2::from([pt.x, pt.y]), FixedVertexHandle::from_index(i))
                })
                .collect_vec();

            handles
                .into_iter()
                .tuple_windows()
                .for_each(|(a, b)| match (a, b) {
                    (Ok(a), Ok(b)) => {
                        if t.can_add_constraint(a, b) {
                            t.add_constraint(a, b);
                        }
                    }
                    _ => {}
                });
        });

        let vertices = t.vertices();
        let faces = t.inner_faces();

        todo!()
    }
}
