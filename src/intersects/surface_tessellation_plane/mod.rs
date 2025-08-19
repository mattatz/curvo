use std::cmp::Ordering;

use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use parry3d_f64::{
    math::UnitVector,
    shape::{FeatureId, Segment},
    utils::{hashmap::HashMap, SortedPair},
};

use crate::{
    misc::{FloatingPoint, Plane},
    prelude::{Intersects, SurfaceTessellation3D},
};

impl<'a, T: FloatingPoint> Intersects<'a, &'a Plane<T>> for SurfaceTessellation3D<T> {
    type Output = anyhow::Result<Vec<Vec<Point3<T>>>>;
    type Option = ();

    /// Find the intersection points between a surface tessellation and a plane
    fn find_intersection(&'a self, other: &'a Plane<T>, _option: Self::Option) -> Self::Output {
        let eps = 1e-10;
        let it = intersection_with_local_plane(&self, &other, T::from_f64(eps).unwrap());
        return it;

        let tri_mesh = parry3d_f64::shape::TriMesh::try_from(self)?;
        let normal = other.normal();
        let axis = UnitVector::new_normalize(Vector3::new(
            normal.x.to_f64().unwrap(),
            normal.y.to_f64().unwrap(),
            normal.z.to_f64().unwrap(),
        ));
        let constant = other.constant().to_f64().unwrap();
        println!(
            "tri_mesh: {:?} {:?}",
            tri_mesh.vertices().len(),
            tri_mesh.indices().len()
        );
        let it = tri_mesh.intersection_with_local_plane(&axis, constant, eps);
        match it {
            parry3d_f64::query::IntersectResult::Intersect(polyline) => {
                let ccs = polyline.extract_connected_components();
                Ok(ccs
                    .into_iter()
                    .map(|cc| {
                        let verts = cc.vertices();
                        let indices = cc.indices();
                        let n = indices.len();
                        println!("n: {}", n);
                        let verts = indices
                            .iter()
                            .map(|index| verts[index[0] as usize])
                            .collect_vec();
                        if n > 0 && indices[0][0] == indices[n - 1][1] {
                            // closed polyline
                            let mut verts = verts.clone();
                            verts.push(verts[0]);
                            verts
                        } else {
                            verts
                        }
                    })
                    .map(|verts| {
                        verts
                            .into_iter()
                            .map(|v| {
                                Point3::new(
                                    T::from_f64(v.x).unwrap(),
                                    T::from_f64(v.y).unwrap(),
                                    T::from_f64(v.z).unwrap(),
                                )
                            })
                            .collect_vec()
                    })
                    .collect())
            }
            _ => Err(anyhow::anyhow!("No intersection found")),
        }
    }
}

fn intersection_with_local_plane<T: FloatingPoint>(
    tess: &SurfaceTessellation3D<T>,
    plane: &Plane<T>,
    epsilon: T,
) -> anyhow::Result<Vec<Vec<Point3<T>>>> {
    // 1. Partition the vertices.
    let vertices = tess
        .points()
        .iter()
        .map(|p| {
            Point3::new(
                p.x.to_f64().unwrap(),
                p.y.to_f64().unwrap(),
                p.z.to_f64().unwrap(),
            )
        })
        .collect_vec();
    let indices = tess.faces();
    let mut colors = vec![0u8; vertices.len()];

    let local_axis = UnitVector::new_normalize(Vector3::new(
        plane.normal().x.to_f64().unwrap(),
        plane.normal().y.to_f64().unwrap(),
        plane.normal().z.to_f64().unwrap(),
    ));
    let bias = plane.constant().to_f64().unwrap();
    let epsilon = epsilon.to_f64().unwrap();

    // Color 0 = on plane.
    //       1 = on negative half-space.
    //       2 = on positive half-space.
    let mut found_negative = false;
    let mut found_positive = false;
    for (i, pt) in vertices.iter().enumerate() {
        let dist_to_plane = pt.coords.dot(&local_axis) - bias;
        if dist_to_plane < -epsilon {
            found_negative = true;
            colors[i] = 1;
        } else if dist_to_plane > epsilon {
            found_positive = true;
            colors[i] = 2;
        }
    }

    // Exit early if `self` isn’t crossed by the plane.
    if !found_negative {
        return Err(anyhow::anyhow!("No intersection found: no negative"));
    }

    if !found_positive {
        return Err(anyhow::anyhow!("No intersection found: no positive"));
    }

    // 2. Split the triangles.
    let mut index_adjacencies: Vec<Vec<usize>> = Vec::new(); // Adjacency list of indices

    // Helper functions for adding polyline segments to the adjacency list
    let mut add_segment_adjacencies = |idx_a: usize, idx_b| {
        assert!(idx_a <= index_adjacencies.len());

        match idx_a.cmp(&index_adjacencies.len()) {
            Ordering::Less => index_adjacencies[idx_a].push(idx_b),
            Ordering::Equal => index_adjacencies.push(vec![idx_b]),
            Ordering::Greater => {}
        }
    };
    let mut add_segment_adjacencies_symmetric = |idx_a: usize, idx_b| {
        if idx_a < idx_b {
            add_segment_adjacencies(idx_a, idx_b);
            add_segment_adjacencies(idx_b, idx_a);
        } else {
            add_segment_adjacencies(idx_b, idx_a);
            add_segment_adjacencies(idx_a, idx_b);
        }
    };

    let mut intersections_found = HashMap::default();
    let mut existing_vertices_found = HashMap::default();
    let mut new_vertices = Vec::new();

    for idx in indices.iter() {
        let mut intersection_features = (FeatureId::Unknown, FeatureId::Unknown);

        // First, find where the plane intersects the triangle.
        for ia in 0..3 {
            let ib = (ia + 1) % 3;
            let idx_a = idx[ia as usize];
            let idx_b = idx[ib as usize];

            let fid = match (colors[idx_a as usize], colors[idx_b as usize]) {
                (1, 2) | (2, 1) => FeatureId::Edge(ia),
                // NOTE: the case (_, 0) will be dealt with in the next loop iteration.
                (0, _) => FeatureId::Vertex(ia),
                _ => continue,
            };

            if intersection_features.0 == FeatureId::Unknown {
                intersection_features.0 = fid;
            } else {
                // TODO: this assertion may fire if the triangle is coplanar with the edge?
                // assert_eq!(intersection_features.1, FeatureId::Unknown);
                intersection_features.1 = fid;
            }
        }

        // Helper that intersects an edge with the plane.
        let mut intersect_edge = |idx_a, idx_b| {
            *intersections_found
                .entry(SortedPair::new(idx_a, idx_b))
                .or_insert_with(|| {
                    let segment = Segment::new(vertices[idx_a as usize], vertices[idx_b as usize]);
                    // Intersect the segment with the plane.
                    if let Some((intersection, _)) = segment
                        .local_split_and_get_intersection(&local_axis, bias, epsilon)
                        .1
                    {
                        new_vertices.push(intersection);
                        colors.push(0);
                        new_vertices.len() - 1
                    } else {
                        unreachable!()
                    }
                })
        };

        // Perform the intersection, push new triangles, and update
        // triangulation constraints if needed.
        match intersection_features {
            (_, FeatureId::Unknown) => {
                // The plane doesn’t intersect the triangle, or intersects it at
                // a single vertex, so we don’t have anything to do.
                assert!(
                    matches!(intersection_features.0, FeatureId::Unknown)
                        || matches!(intersection_features.0, FeatureId::Vertex(_))
                );
            }
            (FeatureId::Vertex(iv1), FeatureId::Vertex(iv2)) => {
                // The plane intersects the triangle along one of its edge.
                // We don’t have to split the triangle, but we need to add
                // the edge to the polyline indices

                let id1 = idx[iv1 as usize];
                let id2 = idx[iv2 as usize];

                let out_id1 = *existing_vertices_found.entry(id1).or_insert_with(|| {
                    let v1 = vertices[id1 as usize];

                    new_vertices.push(v1);
                    new_vertices.len() - 1
                });
                let out_id2 = *existing_vertices_found.entry(id2).or_insert_with(|| {
                    let v2 = vertices[id2 as usize];

                    new_vertices.push(v2);
                    new_vertices.len() - 1
                });

                add_segment_adjacencies_symmetric(out_id1, out_id2);
            }
            (FeatureId::Vertex(iv), FeatureId::Edge(ie))
            | (FeatureId::Edge(ie), FeatureId::Vertex(iv)) => {
                // The plane splits the triangle into exactly two triangles.
                let ia = ie;
                let ib = (ie + 1) % 3;
                let ic = (ie + 2) % 3;
                let idx_a = idx[ia as usize];
                let idx_b = idx[ib as usize];
                let idx_c = idx[ic as usize];
                assert_eq!(iv, ic);

                let intersection_idx = intersect_edge(idx_a, idx_b);

                let out_idx_c = *existing_vertices_found.entry(idx_c).or_insert_with(|| {
                    let v2 = vertices[idx_c as usize];

                    new_vertices.push(v2);
                    new_vertices.len() - 1
                });

                add_segment_adjacencies_symmetric(out_idx_c, intersection_idx);
            }
            (FeatureId::Edge(mut e1), FeatureId::Edge(mut e2)) => {
                // The plane splits the triangle into 1 + 2 triangles.
                // First, make sure the edge indices are consecutive.
                if e2 != (e1 + 1) % 3 {
                    core::mem::swap(&mut e1, &mut e2);
                }

                let ia = e2; // The first point of the second edge is the vertex shared by both edges.
                let ib = (e2 + 1) % 3;
                let ic = (e2 + 2) % 3;
                let idx_a = idx[ia as usize];
                let idx_b = idx[ib as usize];
                let idx_c = idx[ic as usize];

                let intersection1 = intersect_edge(idx_c, idx_a);
                let intersection2 = intersect_edge(idx_a, idx_b);

                add_segment_adjacencies_symmetric(intersection1, intersection2);
            }
            _ => unreachable!(),
        }
    }

    println!("index_adjacencies: {:?}", index_adjacencies.len());

    // todo!()
    Ok(vec![new_vertices.into_iter().map(|v| {
        Point3::new(
            T::from_f64(v.x).unwrap(),
            T::from_f64(v.y).unwrap(),
            T::from_f64(v.z).unwrap(),
        )
    }).collect_vec()])
}
