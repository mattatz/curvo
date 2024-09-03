use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint};

use crate::misc::FloatingPoint;

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
