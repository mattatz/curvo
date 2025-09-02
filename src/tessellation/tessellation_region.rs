use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::{ComplexField, Const, Point2};
use spade::{
    handles::FixedVertexHandle, ConstrainedDelaunayTriangulation, Point2 as SP2, SpadeNum,
    Triangulation,
};

use crate::{
    misc::{FloatingPoint, PolygonBoundary},
    prelude::{Contains, PolygonMesh},
    region::Region,
};

use super::Tessellation;

type Tri<T> = ConstrainedDelaunayTriangulation<SP2<T>>;

impl<T: FloatingPoint + SpadeNum> Tessellation<Option<T>> for Region<T> {
    type Output = anyhow::Result<PolygonMesh<T, Const<2>>>;

    fn tessellate(&self, tolerance: Option<T>) -> Self::Output {
        let mut t = Tri::default();

        let exterior = self.exterior().tessellate(tolerance);
        let interiors = self
            .interiors()
            .iter()
            .map(|c| c.tessellate(tolerance))
            .collect_vec();

        // println!("#{}, #{}", exterior.len(), interiors.len());

        [&[&exterior], interiors.iter().collect_vec().as_slice()]
            .concat()
            .into_iter()
            .for_each(|pts| {
                let handles = pts
                    .iter()
                    .enumerate()
                    .map(|(i, pt)| {
                        t.insert_with_hint(
                            SP2::from([pt.x, pt.y]),
                            FixedVertexHandle::from_index(i),
                        )
                    })
                    .collect_vec();

                handles
                    .into_iter()
                    .circular_tuple_windows()
                    .for_each(|(a, b)| {
                        if let (Ok(a), Ok(b)) = (a, b) {
                            let can_add_constraint = t.can_add_constraint(a, b);
                            // println!("{:?}", can_add_constraint);
                            if can_add_constraint {
                                t.add_constraint(a, b);
                            }
                        }
                    });
            });

        let mut vertices = vec![];

        let vmap: HashMap<_, _> = t
            .vertices()
            .enumerate()
            .map(|(i, v)| {
                let p = v.as_ref();
                vertices.push(Point2::new(p.x, p.y));
                (v.fix(), i)
            })
            .collect();

        let exterior_boundary = PolygonBoundary::new(exterior);
        let interior_boundaries = interiors
            .into_iter()
            .map(|c| PolygonBoundary::new(c))
            .collect_vec();

        let inv_3 = T::from_f64(1. / 3.).unwrap();

        let faces = t
            .inner_faces()
            .filter_map(|f| {
                let vs = f.vertices();
                let tri = vs
                    .iter()
                    .map(|v| v.as_ref())
                    .map(|p| Point2::new(p.x, p.y))
                    .collect_vec();

                let (a, b) = (tri[1] - tri[0], tri[2] - tri[1]);
                let area = a.x * b.y - a.y * b.x;
                if ComplexField::abs(area) < T::default_epsilon() {
                    return None;
                }

                /*
                    let a = vmap[&vs[0].fix()];
                    let b = vmap[&vs[1].fix()];
                    let c = vmap[&vs[2].fix()];
                    return Some([a, b, c]);
                */

                let center: Point2<T> =
                    ((tri[0].coords + tri[1].coords + tri[2].coords) * inv_3).into();
                if exterior_boundary.contains(&center, ()).unwrap()
                    && interior_boundaries
                        .iter()
                        .all(|interior| !interior.contains(&center, ()).unwrap())
                {
                    let a = vmap[&vs[0].fix()];
                    let b = vmap[&vs[1].fix()];
                    let c = vmap[&vs[2].fix()];
                    Some([a, b, c])
                } else {
                    None
                }
            })
            .collect_vec();

        Ok(PolygonMesh::new(vertices, faces))
    }
}
