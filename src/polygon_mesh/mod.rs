use bevy::prelude::Polygon;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OPoint, U2, U3};
use num_traits::Float;

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
