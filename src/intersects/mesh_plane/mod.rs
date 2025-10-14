use std::cmp::Ordering;
use std::collections::HashSet;

use hashbrown::HashMap;
use itertools::Itertools;
use nalgebra::{Point3, U3};
use num_traits::Float;
use simba::scalar::SubsetOf;

use crate::misc::Segment;
use crate::prelude::PolygonMesh;
use crate::{
    misc::{FloatingPoint, Plane},
    prelude::Intersects,
};

#[derive(Debug, Clone)]
pub struct MeshPlaneIntersection<T: FloatingPoint> {
    pub polylines: Vec<Vec<Point3<T>>>,
}

impl<'a, T: FloatingPoint + SubsetOf<f64>> Intersects<'a, &'a Plane<T>> for PolygonMesh<T, U3> {
    type Output = anyhow::Result<MeshPlaneIntersection<T>>;
    type Option = ();

    /// Find the intersection points between a polygon mesh and a plane
    fn find_intersection(&'a self, other: &'a Plane<T>, _option: Self::Option) -> Self::Output {
        let eps = 1e-10;

        intersection_with_local_plane(self, other, T::from_f64(eps).unwrap())
    }
}

fn intersection_with_local_plane<T: FloatingPoint + SubsetOf<f64>>(
    tess: &PolygonMesh<T, U3>,
    plane: &Plane<T>,
    epsilon: T,
) -> anyhow::Result<MeshPlaneIntersection<T>> {
    // 1. Partition the vertices.
    let vertices = tess
        .vertices()
        .iter()
        .map(|p| p.cast::<f64>())
        .collect_vec();
    let indices = tess.faces();
    let mut plane_positions = vec![PlaneSide::On; vertices.len()];

    let plane = plane.cast::<f64>();
    let epsilon = epsilon.to_f64().unwrap();

    let mut found_negative = false;
    let mut found_positive = false;
    for (i, pt) in vertices.iter().enumerate() {
        let dist_to_plane = plane.signed_distance(pt);
        if dist_to_plane < -epsilon {
            found_negative = true;
            plane_positions[i] = PlaneSide::Negative;
        } else if dist_to_plane > epsilon {
            found_positive = true;
            plane_positions[i] = PlaneSide::Positive;
        }
    }

    // Exit early if `self` isn’t crossed by the plane.
    anyhow::ensure!(found_negative, "No intersection found: no negative");
    anyhow::ensure!(found_positive, "No intersection found: no positive");

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

    let mut intersections_found: HashMap<SortedPair<usize>, usize> = HashMap::default();
    let mut existing_vertices_found: HashMap<usize, usize> = HashMap::default();

    let tolerance = 1e-8;
    // let tolerance = tolerance.unwrap_or(1e-4).max(f64::EPSILON);
    let half_tolerance = tolerance * 0.5;
    let exponent = (1. / tolerance).log10();
    let hash_multiplier = 10_f64.powf(exponent);
    let hash_additive = half_tolerance * hash_multiplier;

    let hash = |v: &[f64]| {
        v.iter()
            .map(|c| {
                let c = (c * hash_multiplier + hash_additive).floor();
                c.to_bits()
            })
            .fold(String::new(), |acc, bit| acc + &bit.to_string())
    };

    let mut new_vertices = Vec::new();
    let mut hash_to_index: HashMap<String, usize> = HashMap::default();

    let push_new_vertex = |v: Point3<f64>,
                           hash_to_index: &mut HashMap<String, usize>,
                           new_vertices: &mut Vec<Point3<f64>>|
     -> usize {
        let h = hash(&[v.x, v.y, v.z]);
        if let Some(i) = hash_to_index.get(&h) {
            *i
        } else {
            new_vertices.push(v);
            let idx = new_vertices.len() - 1;
            hash_to_index.insert(h, idx);
            idx
        }
    };

    for idx in indices.iter() {
        let mut intersection_features = (FeatureId::Unknown, FeatureId::Unknown);

        // First, find where the plane intersects the triangle.
        for ia in 0..3 {
            let ib = (ia + 1) % 3;
            let idx_a = idx[ia];
            let idx_b = idx[ib];

            let fid = match (plane_positions[idx_a], plane_positions[idx_b]) {
                (PlaneSide::Negative, PlaneSide::Positive)
                | (PlaneSide::Positive, PlaneSide::Negative) => FeatureId::Edge(ia),
                // NOTE: the case (_, Position::OnPlane) will be dealt with in the next loop iteration.
                (PlaneSide::On, _) => FeatureId::Vertex(ia),
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
        let mut intersect_edge =
            |idx_a,
             idx_b,
             hash_to_index: &mut HashMap<String, usize>,
             new_vertices: &mut Vec<Point3<f64>>| {
                *intersections_found
                    .entry(SortedPair::new(idx_a, idx_b))
                    .or_insert_with(|| {
                        let segment = Segment::new(vertices[idx_a], vertices[idx_b]);
                        // Intersect the segment with the plane.
                        if let Some((intersection, _)) = segment.split(&plane, epsilon).1 {
                            push_new_vertex(intersection, hash_to_index, new_vertices)
                        } else {
                            /*
                            let dist = plane.signed_distance(&vertices[idx_a]);
                            let dist2 = plane.signed_distance(&vertices[idx_b]);
                            println!("dist: {:?}, dist2: {:?}", dist, dist2);
                            */
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

                let id1 = idx[iv1];
                let id2 = idx[iv2];

                let out_id1 = *existing_vertices_found.entry(id1).or_insert_with(|| {
                    let v1 = vertices[id1];
                    push_new_vertex(v1, &mut hash_to_index, &mut new_vertices)
                });
                let out_id2 = *existing_vertices_found.entry(id2).or_insert_with(|| {
                    let v2 = vertices[id2];
                    push_new_vertex(v2, &mut hash_to_index, &mut new_vertices)
                });

                add_segment_adjacencies_symmetric(out_id1, out_id2);
            }
            (FeatureId::Vertex(iv), FeatureId::Edge(ie))
            | (FeatureId::Edge(ie), FeatureId::Vertex(iv)) => {
                // The plane splits the triangle into exactly two triangles.
                let ia = ie;
                let ib = (ie + 1) % 3;
                let ic = (ie + 2) % 3;
                let idx_a = idx[ia];
                let idx_b = idx[ib];
                let idx_c = idx[ic];
                assert_eq!(iv, ic);

                let intersection_idx =
                    intersect_edge(idx_a, idx_b, &mut hash_to_index, &mut new_vertices);

                let out_idx_c = *existing_vertices_found.entry(idx_c).or_insert_with(|| {
                    let v2 = vertices[idx_c];
                    push_new_vertex(v2, &mut hash_to_index, &mut new_vertices)
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
                let idx_a = idx[ia];
                let idx_b = idx[ib];
                let idx_c = idx[ic];

                let intersection1 =
                    intersect_edge(idx_c, idx_a, &mut hash_to_index, &mut new_vertices);
                let intersection2 =
                    intersect_edge(idx_a, idx_b, &mut hash_to_index, &mut new_vertices);

                add_segment_adjacencies_symmetric(intersection1, intersection2);
            }
            _ => unreachable!(),
        }
    }

    /*
    let polylines = index_adjacencies
        .iter()
        .enumerate()
        .flat_map(|(i, adj)| {
            adj.iter()
                .map(|j| {
                    // println!("{} -> {}", i, dx);
                    vec![new_vertices[i], new_vertices[*j]]
                })
                .collect_vec()
        })
        .collect_vec();

    Ok(MeshPlaneIntersection {
        polylines: polylines
            .into_iter()
            .map(|polyline| {
                polyline
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
            .collect_vec(),
    })
    */

    /*
        let mut polylines = vec![];
        let mut set: HashSet<(usize, usize)> = HashSet::default();
        index_adjacencies.iter().enumerate().for_each(|(i, adj)| {
            adj.iter().for_each(|j| {
                let k = if i < *j { (i, *j) } else { (*j, i) };
                if !set.contains(&k) {
                    set.insert(k);
                    polylines.push(vec![i, *j]);
                }
            });
        });
        return Ok(MeshPlaneIntersection {
            polylines: polylines
                .into_iter()
                .map(|polyline| {
                    polyline
                        .into_iter()
                        .map(|i| {
                            let v = new_vertices[i];
                            Point3::new(
                                T::from_f64(v.x).unwrap(),
                                T::from_f64(v.y).unwrap(),
                                T::from_f64(v.z).unwrap(),
                            )
                        })
                        .collect_vec()
                })
                .collect_vec(),
        });
    */

    /*
    println!("vertices: {}", new_vertices.len());
    let n = index_adjacencies.iter().filter(|adj| adj.len() < 2).count();
    let m = index_adjacencies
        .iter()
        .filter(|adj| adj.len() >= 2)
        .count();
    println!("n: {}, m: {}", n, m);
    */

    // Build polylines from the adjacency list
    let polylines = extract_polylines_from_adjacencies(&index_adjacencies)
        .into_iter()
        .map(|polyline| {
            polyline
                .to_points(&new_vertices)
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
        .collect_vec();

    Ok(MeshPlaneIntersection { polylines })
}

/// Polyline is a collection of indices of vertices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Polyline {
    Open(Vec<usize>),   // Open polyline
    Closed(Vec<usize>), // Closed polyline
}

impl Polyline {
    pub fn to_points(&self, vertices: &[Point3<f64>]) -> Vec<Point3<f64>> {
        match self {
            Polyline::Open(indices) => indices.iter().map(|i| vertices[*i]).collect_vec(),
            Polyline::Closed(indices) => indices.iter().map(|i| vertices[*i]).collect_vec(),
        }
    }
}

/// Extract polylines from an adjacency list.
/// - Input: adj[i] is the list of vertices adjacent to vertex i (can be one-way).
/// - Output: maximum polyline that does not cross branches (one per component if there are no branches).
pub fn extract_polylines_from_adjacencies(adjacencies: &[Vec<usize>]) -> Vec<Polyline> {
    let n = adjacencies.len();

    // 1) Undirected + deduplication
    let mut undirected = vec![Vec::<usize>::new(); n];
    for u in 0..n {
        for &v in &adjacencies[u] {
            if v < n {
                undirected[u].push(v);
                undirected[v].push(u);
            }
        }
    }
    for nbrs in &mut undirected {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    let deg: Vec<usize> = undirected.iter().map(|v| v.len()).collect();

    // Record used edges (undirected, so managed by (min,max)).
    let mut used: HashSet<(usize, usize)> = HashSet::new();
    let mut polylines: Vec<Polyline> = Vec::new();

    let edge_key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let mark = |a: usize, b: usize, used: &mut HashSet<(usize, usize)>| {
        used.insert(edge_key(a, b));
    };
    let edge_used =
        |a: usize, b: usize, used: &HashSet<(usize, usize)>| used.contains(&edge_key(a, b));

    // 2) Extend maximum polylines from branches (deg != 2) and endpoints (deg == 1).
    for u in 0..n {
        let du = deg[u];
        if du == 0 {
            // Isolated points are returned as open polylines of length 0.
            polylines.push(Polyline::Open(vec![u]));
            continue;
        }
        if du != 2 {
            for &v in &undirected[u] {
                if edge_used(u, v, &used) {
                    continue;
                }
                // Follow a straight line (deg==2) in the u -> v direction.
                let mut path = Vec::new();
                path.push(u);
                path.push(v);
                mark(u, v, &mut used);

                let mut prev = u;
                let mut cur = v;
                // Follow a straight line (deg==2) in the u -> v direction.
                while undirected[cur].len() == 2 {
                    let a = undirected[cur][0];
                    let b = undirected[cur][1];
                    let next = if a == prev { b } else { a };
                    if edge_used(cur, next, &used) {
                        break; // Encountered an already visited edge (safety).
                    }
                    mark(cur, next, &mut used);
                    path.push(next);
                    prev = cur;
                    cur = next;
                }

                polylines.push(Polyline::Open(path));
            }
        }
    }

    // 3) Remaining edges are closed loops (2-regular components), so collect them as loops.
    for s in 0..n {
        if deg[s] == 2 {
            for &v in &undirected[s] {
                if edge_used(s, v, &used) {
                    continue;
                }

                // Follow a loop from s -> v.
                let mut cycle = Vec::new();
                cycle.push(s);
                mark(s, v, &mut used);
                let mut prev = s;
                let mut cur = v;

                while cur != s {
                    cycle.push(cur);
                    // cur must be deg==2.
                    let a = undirected[cur][0];
                    let b = undirected[cur][1];
                    let next = if a == prev { b } else { a };
                    if edge_used(cur, next, &used) && next != s {
                        break; // It should be impossible, but guard.
                    }
                    mark(cur, next, &mut used);
                    prev = cur;
                    cur = next;
                }

                polylines.push(Polyline::Closed(cycle));
            }
        }
    }

    polylines
}

/// The position of a point relative to a plane.
#[derive(Copy, Clone, Debug, PartialEq)]
enum PlaneSide {
    On,
    Negative,
    Positive,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
enum FeatureId {
    /// Shape-dependent identifier of a vertex.
    Vertex(usize),
    /// Shape-dependent identifier of an edge.
    Edge(usize),
    #[default]
    Unknown,
}

/// A pair of elements sorted in increasing order.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SortedPair<T: PartialOrd>([T; 2]);

impl<T: PartialOrd> SortedPair<T> {
    /// Sorts two elements in increasing order into a new pair.
    pub fn new(element1: T, element2: T) -> Self {
        if element1 > element2 {
            SortedPair([element2, element1])
        } else {
            SortedPair([element1, element2])
        }
    }
}

#[cfg(test)]
mod tests {}
