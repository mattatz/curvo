use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, U2, U3};

use crate::{misc::FloatingPoint, prelude::SurfaceTessellation3D};

/// A struct representing a polygon mesh.
#[derive(Clone, Debug)]
pub struct PolygonMesh<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    vertices: Vec<OPoint<T, D>>,
    faces: Vec<[usize; 3]>,
}

impl<T: FloatingPoint, D: DimName> PolygonMesh<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(vertices: Vec<OPoint<T, D>>, faces: Vec<[usize; 3]>) -> Self {
        Self { vertices, faces }
    }

    pub fn vertices(&self) -> &[OPoint<T, D>] {
        &self.vertices
    }

    pub fn faces(&self) -> &[[usize; 3]] {
        &self.faces
    }

    pub fn triangles(&self) -> Vec<[OPoint<T, D>; 3]> {
        self.faces
            .iter()
            .map(|[a, b, c]| {
                [
                    self.vertices[*a].clone(),
                    self.vertices[*b].clone(),
                    self.vertices[*c].clone(),
                ]
            })
            .collect()
    }
}

impl<T: FloatingPoint> PolygonMesh<T, U2> {
    pub fn area(&self) -> T {
        let triangles = self.triangles();
        triangles
            .iter()
            .map(|[a, b, c]| {
                let ab = b - a;
                let ac = c - a;
                (ab.x * ac.y - ab.y * ac.x).abs()
            })
            .fold(T::zero(), |a, b| a + b)
            / (T::from_usize(2).unwrap())
    }
}

impl<T: FloatingPoint> PolygonMesh<T, U3> {
    pub fn area(&self) -> T {
        let triangles = self.triangles();
        triangles
            .iter()
            .map(|[a, b, c]| {
                let ab = b - a;
                let ac = c - a;
                let cross = ab.cross(&ac);
                cross.norm()
            })
            .fold(T::zero(), |a, b| a + b)
            / (T::from_usize(2).unwrap())
    }
}

/// Merge two polygon meshes
impl<T: FloatingPoint, D: DimName> std::ops::Add<PolygonMesh<T, D>> for PolygonMesh<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    type Output = PolygonMesh<T, D>;

    fn add(self, rhs: PolygonMesh<T, D>) -> Self::Output {
        let v0 = self.vertices.len();
        let vertices = [self.vertices, rhs.vertices].concat();
        let faces = [
            self.faces,
            rhs.faces
                .iter()
                .map(|[a, b, c]| [*a + v0, *b + v0, *c + v0])
                .collect(),
        ]
        .concat();
        Self::Output::new(vertices, faces)
    }
}

/// Merge multiple polygon meshes
impl<T: FloatingPoint, D: DimName> std::iter::Sum for PolygonMesh<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(vec![], vec![]), |a, b| a + b)
    }
}

/// Convert a surface tessellation to a polygon mesh
impl<T: FloatingPoint> From<SurfaceTessellation3D<T>> for PolygonMesh<T, U3> {
    fn from(tess: SurfaceTessellation3D<T>) -> Self {
        Self::new(tess.points().to_vec(), tess.faces().to_vec())
    }
}
