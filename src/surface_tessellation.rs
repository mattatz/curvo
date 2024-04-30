use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector,
    Vector2, U1,
};

use crate::{
    adaptive_tessellation_node::AdaptiveTessellationNode, prelude::NurbsSurface, FloatingPoint,
};

/// Surface tessellation representation
/// This struct is used to create a mesh data from surface
#[derive(Clone, Debug)]
pub struct SurfaceTessellation<T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    pub(crate) points: Vec<OPoint<T, DimNameDiff<D, U1>>>,
    pub(crate) normals: Vec<OVector<T, DimNameDiff<D, U1>>>,
    pub(crate) faces: Vec<[usize; 3]>,
    pub(crate) uvs: Vec<Vector2<T>>,
}

impl<T: FloatingPoint, D: DimName> SurfaceTessellation<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    /// Create a new surface tessellation from surface and adaptive tessellation nodes
    pub fn new(surface: &NurbsSurface<T, D>, nodes: &Vec<AdaptiveTessellationNode<T, D>>) -> Self {
        let mut tess = Self {
            points: Default::default(),
            normals: Default::default(),
            faces: Default::default(),
            uvs: Default::default(),
        };

        nodes.iter().for_each(|node| {
            tess.triangulate(surface, nodes, node);
        });

        tess
    }

    /// Triangulate the surface with adaptive tessellation nodes recursively
    fn triangulate(
        &mut self,
        surface: &NurbsSurface<T, D>,
        nodes: &Vec<AdaptiveTessellationNode<T, D>>,
        node: &AdaptiveTessellationNode<T, D>,
    ) {
        if node.is_leaf() {
            let mut base_index = self.points.len();
            let mut uvs = vec![];
            let mut ids = vec![];
            let mut split_id = 0;
            for i in 0..4 {
                let edge_corners = node.get_all_corners(nodes, i);
                if edge_corners.len() == 2 {
                    split_id = i + 1;
                }

                uvs.extend(edge_corners);
            }

            uvs.iter().for_each(|corner| {
                self.points.push(corner.point.clone());
                self.normals.push(corner.normal.clone());
                self.uvs.push(corner.uv);
                ids.push(base_index);
                base_index += 1;
            });

            match uvs.len() {
                4 => {
                    self.faces.push([ids[0], ids[3], ids[1]]);
                    self.faces.push([ids[3], ids[2], ids[1]]);
                }
                5 => {
                    let il = ids.len();
                    self.faces.push([
                        ids[split_id],
                        ids[(split_id + 2) % il],
                        ids[(split_id + 1) % il],
                    ]);
                    self.faces.push([
                        ids[(split_id + 4) % il],
                        ids[(split_id + 3) % il],
                        ids[split_id],
                    ]);
                    self.faces.push([
                        ids[split_id],
                        ids[(split_id + 3) % il],
                        ids[(split_id + 2) % il],
                    ]);
                }
                n => {
                    let center = node.center(surface);
                    self.points.push(center.point.clone());
                    self.normals.push(center.normal.clone());
                    self.uvs.push(center.uv);

                    let center_index = self.points.len() - 1;
                    let mut j = n - 1;
                    let mut i = 0;
                    while i < n {
                        self.faces.push([center_index, ids[i], ids[j]]);
                        j = i;
                        i += 1;
                    }
                }
            };
        } else {
            node.children.iter().for_each(|child| {
                let c = &nodes[*child];
                self.triangulate(surface, nodes, c);
            });
        }
    }

    pub fn points(&self) -> &Vec<OPoint<T, DimNameDiff<D, U1>>> {
        &self.points
    }

    pub fn normals(&self) -> &Vec<OVector<T, DimNameDiff<D, U1>>> {
        &self.normals
    }

    pub fn uvs(&self) -> &Vec<Vector2<T>> {
        &self.uvs
    }

    pub fn faces(&self) -> &Vec<[usize; 3]> {
        &self.faces
    }
}
