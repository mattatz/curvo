use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator, DimName, OPoint, Point2};
use spade::{ConstrainedDelaunayTriangulation, Point2 as SP2, SpadeNum, Triangulation};

use crate::prelude::{Contains, PolygonMesh, Tessellation};

use super::{orientation, FloatingPoint, Orientation};

/// A boundary of a polygon curve.
#[derive(Debug, Clone)]
pub struct PolygonBoundary<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    vertices: Vec<OPoint<T, D>>,
}

impl<T: FloatingPoint, D: DimName> PolygonBoundary<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(vertices: Vec<OPoint<T, D>>) -> Self {
        Self { vertices }
    }

    pub fn vertices(&self) -> &Vec<OPoint<T, D>> {
        &self.vertices
    }
}

impl<T: FloatingPoint, D: DimName> FromIterator<OPoint<T, D>> for PolygonBoundary<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn from_iter<I: IntoIterator<Item = OPoint<T, D>>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

/// Check if a point is contained in a polygon curve.
/// ```
/// use nalgebra::Point2;
/// use curvo::prelude::PolygonBoundary;
/// use curvo::prelude::Contains;
/// let boundary = vec![
///   Point2::new(0., 0.),
///   Point2::new(1., 0.),
///   Point2::new(1., 1.),
///   Point2::new(0., 1.),
/// ];
/// let curve = PolygonBoundary::new(boundary);
/// assert!(curve.contains(&Point2::new(0.5, 0.5), ()).unwrap());
/// assert!(!curve.contains(&Point2::new(0.5, 1.5), ()).unwrap());
/// ```
impl<T: FloatingPoint> Contains<OPoint<T, Const<2>>> for PolygonBoundary<T, Const<2>> {
    type Option = ();
    fn contains(&self, c: &Point2<T>, _option: Self::Option) -> anyhow::Result<bool> {
        let winding_number = self.vertices().iter().circular_tuple_windows().fold(
            0_i32,
            move |winding_number, (p0, p1)| {
                if p0.y <= c.y {
                    if p1.y >= c.y {
                        let o = orientation(p0, p1, c);
                        if o == Orientation::CounterClockwise && p1.y != c.y {
                            return winding_number + 1;
                        }
                    }
                } else if p1.y <= c.y {
                    let o = orientation(p0, p1, c);
                    if o == Orientation::Clockwise {
                        return winding_number - 1;
                    }
                }
                winding_number
            },
        );
        Ok(winding_number != 0)
    }
}

impl<T: FloatingPoint + SpadeNum> Tessellation<()> for PolygonBoundary<T, Const<2>> {
    type Output = anyhow::Result<PolygonMesh<T, Const<2>>>;

    /// Tessellate the polygon boundary into a polygon mesh
    fn tessellate(&self, _options: ()) -> Self::Output {
        let mut cdt = ConstrainedDelaunayTriangulation::<SP2<T>>::default();

        // Insert boundary points and build constraint edges
        let mut vertex_handles = Vec::new();
        for point in self.vertices() {
            let spade_point = SP2::new(point.x, point.y);
            let handle = cdt.insert(spade_point)?;
            vertex_handles.push(handle);
        }

        // Add constraint edges to form the boundary
        for i in 0..vertex_handles.len() {
            let next = (i + 1) % vertex_handles.len();
            cdt.add_constraint(vertex_handles[i], vertex_handles[next]);
        }

        // Extract triangles and convert to 3D
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        let mut vertex_map = std::collections::HashMap::new();

        for face in cdt.inner_faces() {
            let [v0, v1, v2] = face.vertices();

            let mut face_indices = Vec::new();
            for vertex in [v0, v1, v2] {
                let idx = if let Some(&existing_idx) = vertex_map.get(&vertex.fix()) {
                    existing_idx
                } else {
                    let pos = vertex.position();
                    let new_idx = vertices.len();
                    vertices.push(Point2::new(pos.x, pos.y));
                    vertex_map.insert(vertex.fix(), new_idx);
                    new_idx
                };
                face_indices.push(idx);
            }

            faces.push([face_indices[0], face_indices[1], face_indices[2]]);
        }

        Ok(PolygonMesh::new(vertices, faces))
    }
}
