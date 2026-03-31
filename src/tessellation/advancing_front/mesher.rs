use itertools::Itertools;
use nalgebra::{Point2, Point3, Vector2, Vector3};
use spade::{ConstrainedDelaunayTriangulation, SpadeNum, Triangulation};

use crate::misc::FloatingPoint;
use crate::prelude::SurfaceTessellation3D;
use crate::region::CompoundCurve2D;
use crate::tessellation::surface_metric::{curvature_to_edge_length, SurfaceMetric};
use crate::tessellation::trimmed_surface::trimmed_surface_ext::TrimmedSurfaceExt;
use crate::tessellation::trimmed_surface::Vertex;

use super::options::AdvancingFrontOptions;

/// Advancing front mesher for trimmed parametric surfaces.
///
/// Instead of relying on a CDT with UV-space Delaunay criterion,
/// this mesher generates triangles by advancing inward from the trim boundary,
/// using the surface metric tensor to control triangle size based on 3D curvature.
pub struct AdvancingFrontMesher<'a, T: FloatingPoint, S> {
    surface: &'a S,
    options: AdvancingFrontOptions<T>,
    /// All generated vertices: (point, normal, uv)
    points: Vec<Point3<T>>,
    normals: Vec<Vector3<T>>,
    uvs: Vec<Vector2<T>>,
    /// Generated triangle faces (indices into points/normals/uvs)
    faces: Vec<[usize; 3]>,
}

impl<'a, T, S> AdvancingFrontMesher<'a, T, S>
where
    T: FloatingPoint + SpadeNum,
    S: TrimmedSurfaceExt<
            T,
            fn(
                &crate::prelude::AdaptiveTessellationNode<T, nalgebra::U4>,
            ) -> Option<crate::prelude::DividableDirection>,
        > + Sync,
{
    pub fn new(surface: &'a S, options: AdvancingFrontOptions<T>) -> Self {
        Self {
            surface,
            options,
            points: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Run the mesher and return the tessellation result.
    pub fn mesh(mut self) -> anyhow::Result<SurfaceTessellation3D<T>> {
        // Step 1: Discretize boundary curves with curvature-adaptive sampling
        let exterior_pts = self
            .surface
            .exterior()
            .map(|curve| self.discretize_boundary(curve))
            .transpose()?;
        let interior_pts: Vec<Vec<usize>> = self
            .surface
            .interiors()
            .iter()
            .map(|curve| self.discretize_boundary(curve))
            .collect::<anyhow::Result<Vec<_>>>()?;

        // Step 2: Build CDT from boundary points as initial triangulation
        let mut cdt = ConstrainedDelaunayTriangulation::<Vertex<T>>::default();

        // Insert all boundary vertices
        let handles: Vec<_> = self
            .uvs
            .iter()
            .zip(self.points.iter())
            .zip(self.normals.iter())
            .map(|((uv, p), n)| {
                let v = Vertex::new(*p, *n, *uv);
                cdt.insert(v)
            })
            .collect::<Vec<_>>();

        // Add boundary constraint edges
        let add_constraints = |cdt: &mut ConstrainedDelaunayTriangulation<Vertex<T>>,
                               indices: &[usize]| {
            for (a, b) in indices.iter().circular_tuple_windows() {
                if let (Ok(ha), Ok(hb)) = (&handles[*a], &handles[*b]) {
                    if cdt.can_add_constraint(*ha, *hb) {
                        cdt.add_constraint(*ha, *hb);
                    }
                }
            }
        };

        if let Some(ref ext) = exterior_pts {
            add_constraints(&mut cdt, ext);
        }
        for interior in &interior_pts {
            add_constraints(&mut cdt, interior);
        }

        // Step 3: Insert interior points based on curvature-adaptive spacing
        self.insert_interior_points(&mut cdt, &handles)?;

        // Step 4: Extract triangles inside the trim region
        self.extract_trimmed_faces(&cdt, &exterior_pts, &interior_pts)?;

        Ok(SurfaceTessellation3D::raw(
            self.points,
            self.normals,
            self.uvs,
            self.faces,
        ))
    }

    /// Discretize a boundary curve with curvature-adaptive sampling.
    fn discretize_boundary(&mut self, curve: &CompoundCurve2D<T>) -> anyhow::Result<Vec<usize>> {
        let mut indices = Vec::new();
        let eps = T::from_f64(1e-8).unwrap();

        for (i, span) in curve.spans().iter().enumerate() {
            let (t_start, t_end) = span.knots_domain();
            let pts = self.adaptive_discretize_curve(span, t_start, t_end, 0);

            for (j, uv) in pts.into_iter().enumerate() {
                // Skip first point of subsequent spans (shared with previous)
                if i > 0 && j == 0 {
                    continue;
                }
                let idx = self.add_surface_vertex(uv);
                indices.push(idx);
            }
        }

        // Close the boundary: skip last if it matches first
        if indices.len() > 1 {
            let first_uv = self.uvs[indices[0]];
            let last_uv = self.uvs[*indices.last().unwrap()];
            if (first_uv - last_uv).norm() < eps {
                indices.pop();
            }
        }

        Ok(indices)
    }

    /// Adaptively discretize a 2D curve on the surface using chord-height criterion.
    fn adaptive_discretize_curve(
        &self,
        curve: &crate::curve::NurbsCurve2D<T>,
        t_start: T,
        t_end: T,
        depth: usize,
    ) -> Vec<Vector2<T>> {
        let max_depth = 10;
        let half = T::from_f64(0.5).unwrap();

        let uv_start = curve.point_at(t_start);
        let uv_end = curve.point_at(t_end);
        let t_mid = t_start + (t_end - t_start) * half;
        let uv_mid = curve.point_at(t_mid);

        // 3D positions
        let p_start = self.surface.point_at(uv_start.x, uv_start.y);
        let p_end = self.surface.point_at(uv_end.x, uv_end.y);
        let p_mid = self.surface.point_at(uv_mid.x, uv_mid.y);

        // Chord-height: distance from surface midpoint to linear midpoint
        let linear_mid = (p_start.coords + p_end.coords) * half;
        let deviation = (p_mid.coords - linear_mid).norm();

        // Also check 3D edge length vs max_edge_length
        let edge_len = (p_end - p_start).norm();

        let needs_split = depth < max_depth
            && (deviation > self.options.deflection || edge_len > self.options.max_edge_length);

        if needs_split {
            let mut left = self.adaptive_discretize_curve(curve, t_start, t_mid, depth + 1);
            let right = self.adaptive_discretize_curve(curve, t_mid, t_end, depth + 1);
            left.pop(); // remove duplicate midpoint
            left.extend(right);
            left
        } else {
            vec![uv_start.coords, uv_end.coords]
        }
    }

    /// Insert interior points into the CDT using locally adaptive curvature-based spacing.
    /// Walks along the U direction, computing local metric at each step to determine
    /// the next point's position. This produces denser points where curvature is high.
    ///
    /// When the `rayon` feature is enabled, surface evaluations are parallelized.
    fn insert_interior_points(
        &mut self,
        cdt: &mut ConstrainedDelaunayTriangulation<Vertex<T>>,
        _handles: &[Result<spade::handles::FixedVertexHandle, spade::InsertionError>],
    ) -> anyhow::Result<()> {
        let ((u_min, u_max), (v_min, v_max)) = self.surface.knots_domain();
        let eps = T::from_f64(1e-6).unwrap();

        // First pass: determine V-direction row positions using locally adaptive spacing
        let v_positions = self.adaptive_parameter_steps(v_min + eps, v_max - eps, |v| {
            let u_mid = (u_min + u_max) * T::from_f64(0.5).unwrap();
            self.target_uv_step_at(u_mid, v).y
        });

        // Second pass: for each V row, determine U positions
        let mut uv_grid: Vec<Vector2<T>> = Vec::new();
        for &v in &v_positions {
            let u_positions = self.adaptive_parameter_steps(u_min + eps, u_max - eps, |u| {
                self.target_uv_step_at(u, v).x
            });
            for &u in &u_positions {
                uv_grid.push(Vector2::new(u, v));
            }
        }

        // Evaluate surface at all interior UV positions (parallelized when rayon is enabled)
        let evaluated = self.evaluate_points_batch(&uv_grid);

        // Insert into CDT sequentially (spade requirement)
        for (uv, p, n) in evaluated {
            let vertex = Vertex::new(p, n, uv);
            let _ = cdt.insert(vertex);
            self.points.push(p);
            self.normals.push(n);
            self.uvs.push(uv);
        }

        Ok(())
    }

    /// Evaluate surface point and normal at multiple UV positions.
    /// Parallelized with rayon when the feature is enabled.
    fn evaluate_points_batch(
        &self,
        uv_positions: &[Vector2<T>],
    ) -> Vec<(Vector2<T>, Point3<T>, Vector3<T>)> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            uv_positions
                .par_iter()
                .map(|uv| {
                    let p = self.surface.point_at(uv.x, uv.y);
                    let n = self.surface.normal_at(uv.x, uv.y);
                    (*uv, p, n)
                })
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            uv_positions
                .iter()
                .map(|uv| {
                    let p = self.surface.point_at(uv.x, uv.y);
                    let n = self.surface.normal_at(uv.x, uv.y);
                    (*uv, p, n)
                })
                .collect()
        }
    }

    /// Generate parameter values with locally adaptive spacing.
    /// `step_fn(t)` returns the desired step size at parameter `t`.
    fn adaptive_parameter_steps(&self, t_min: T, t_max: T, step_fn: impl Fn(T) -> T) -> Vec<T> {
        let mut positions = Vec::new();
        let mut t = t_min;
        let max_iters = 1000; // safety limit
        let mut iters = 0;

        while t < t_max && iters < max_iters {
            positions.push(t);
            let step = step_fn(t).max(self.options.min_edge_length);
            t += step;
            iters += 1;
        }

        // Don't include the boundary itself (boundary points already exist)
        positions
            .into_iter()
            .filter(|&t| {
                t > t_min + self.options.min_edge_length && t < t_max - self.options.min_edge_length
            })
            .collect()
    }

    /// Compute target UV step size at a point based on local metric and curvature.
    fn target_uv_step_at(&self, u: T, v: T) -> Vector2<T> {
        let target_3d = self.target_edge_length_at(u, v);
        let metric = self.compute_metric(u, v);
        metric.max_uv_step(target_3d)
    }

    /// Compute target 3D edge length at a UV point based on surface curvature.
    fn target_edge_length_at(&self, u: T, v: T) -> T {
        let eps = T::from_f64(1e-6).unwrap();
        let h = T::from_f64(1e-4).unwrap();

        // Estimate principal curvatures via finite differences of normals
        let n0 = self.surface.normal_at(u, v);

        let ((_u_min, u_max), (_v_min, v_max)) = self.surface.knots_domain();
        let u_h = (u + h).min(u_max - eps);
        let v_h = (v + h).min(v_max - eps);

        let n_du = self.surface.normal_at(u_h, v);
        let n_dv = self.surface.normal_at(u, v_h);

        let su = self.surface.point_at(u_h, v) - self.surface.point_at(u, v);
        let sv = self.surface.point_at(u, v_h) - self.surface.point_at(u, v);

        let su_len = su.norm();
        let sv_len = sv.norm();

        // Approximate curvature as rate of normal change per arc length
        let k_u = if su_len > eps {
            (n_du - n0).norm() / su_len
        } else {
            T::zero()
        };
        let k_v = if sv_len > eps {
            (n_dv - n0).norm() / sv_len
        } else {
            T::zero()
        };

        let k_max = k_u.max(k_v);
        let target = curvature_to_edge_length(k_max, self.options.deflection);
        target
            .max(self.options.min_edge_length)
            .min(self.options.max_edge_length)
    }

    /// Compute the surface metric (first fundamental form) at a UV point.
    fn compute_metric(&self, u: T, v: T) -> SurfaceMetric<T> {
        let eps = T::from_f64(1e-6).unwrap();
        let h = T::from_f64(1e-5).unwrap();

        let ((_u_min, u_max), (_v_min, v_max)) = self.surface.knots_domain();
        let u_h = (u + h).min(u_max - eps);
        let v_h = (v + h).min(v_max - eps);

        let p = self.surface.point_at(u, v);
        let pu = self.surface.point_at(u_h, v);
        let pv = self.surface.point_at(u, v_h);

        let su = (pu - p) / (u_h - u);
        let sv = (pv - p) / (v_h - v);

        let e = su.dot(&su);
        let g = sv.dot(&sv);

        SurfaceMetric::new(e, g)
    }

    /// Extract faces from the CDT that are inside the trim region.
    /// Contains-check is parallelized with rayon when available.
    fn extract_trimmed_faces(
        &mut self,
        cdt: &ConstrainedDelaunayTriangulation<Vertex<T>>,
        exterior: &Option<Vec<usize>>,
        interiors: &[Vec<usize>],
    ) -> anyhow::Result<()> {
        use crate::misc::PolygonBoundary;
        use crate::prelude::Contains;
        use nalgebra::ComplexField;

        let inv_3 = T::from_f64(1. / 3.).unwrap();
        let half = T::from_f64(0.5).unwrap();
        let shrink = T::from_f64(0.01).unwrap();

        let uv_exterior = exterior.as_ref().map(|ext| {
            PolygonBoundary::new(ext.iter().map(|&i| Point2::from(self.uvs[i])).collect())
        });

        let uv_interiors: Vec<_> = interiors
            .iter()
            .map(|int| {
                PolygonBoundary::new(int.iter().map(|&i| Point2::from(self.uvs[i])).collect())
            })
            .collect();

        // Build vertex index map: CDT vertex handle → sequential index
        let mut vmap = std::collections::HashMap::new();
        for (i, v) in cdt.vertices().enumerate() {
            vmap.insert(v.fix(), i);
        }

        let cdt_verts: Vec<_> = cdt.vertices().collect();

        // Phase 1: collect candidate faces (sequential CDT iteration)
        struct CandidateFace<T> {
            tri_uvs: [nalgebra::Vector2<T>; 3],
            cdt_indices: [usize; 3],
        }

        let mut candidates: Vec<CandidateFace<T>> = Vec::new();

        for face in cdt.inner_faces() {
            let vs = face.vertices();
            let tri_uvs: [Vector2<T>; 3] = [
                vs[0].as_ref().uv(),
                vs[1].as_ref().uv(),
                vs[2].as_ref().uv(),
            ];

            let (a, b) = (tri_uvs[1] - tri_uvs[0], tri_uvs[2] - tri_uvs[1]);
            let area = a.x * b.y - a.y * b.x;
            if ComplexField::abs(area) < T::default_epsilon() {
                continue;
            }

            let cdt_indices = [vmap[&vs[0].fix()], vmap[&vs[1].fix()], vmap[&vs[2].fix()]];

            candidates.push(CandidateFace {
                tri_uvs,
                cdt_indices,
            });
        }

        // Phase 2: contains-check (parallelized when rayon is available)
        let is_inside = |tri_uvs: &[Vector2<T>; 3]| -> bool {
            let center: Point2<T> = ((tri_uvs[0] + tri_uvs[1] + tri_uvs[2]) * inv_3).into();
            let mid_01: Point2<T> = ((tri_uvs[0] + tri_uvs[1]) * half * (T::one() - shrink)
                + (tri_uvs[2]) * shrink)
                .into();
            let mid_12: Point2<T> = ((tri_uvs[1] + tri_uvs[2]) * half * (T::one() - shrink)
                + (tri_uvs[0]) * shrink)
                .into();
            let mid_20: Point2<T> = ((tri_uvs[2] + tri_uvs[0]) * half * (T::one() - shrink)
                + (tri_uvs[1]) * shrink)
                .into();

            [center, mid_01, mid_12, mid_20].iter().all(|pt| {
                let in_ext = uv_exterior
                    .as_ref()
                    .map(|ext| ext.contains(pt, ()).unwrap_or(false))
                    .unwrap_or(true);
                let in_int = !uv_interiors.is_empty()
                    && uv_interiors
                        .iter()
                        .any(|int| int.contains(pt, ()).unwrap_or(false));
                in_ext && !in_int
            })
        };

        #[cfg(feature = "rayon")]
        let inside_flags: Vec<bool> = {
            use rayon::prelude::*;
            candidates
                .par_iter()
                .map(|c| is_inside(&c.tri_uvs))
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let inside_flags: Vec<bool> = candidates.iter().map(|c| is_inside(&c.tri_uvs)).collect();

        // Phase 3: add accepted faces (sequential vertex remapping)
        for (candidate, &inside) in candidates.iter().zip(inside_flags.iter()) {
            if !inside {
                continue;
            }
            let indices: Option<[usize; 3]> = candidate
                .cdt_indices
                .iter()
                .map(|&cdt_idx| {
                    let vert = cdt_verts[cdt_idx].as_ref();
                    self.find_or_add_vertex(vert.point(), vert.normal(), vert.uv())
                })
                .collect_array::<3>();
            if let Some(face) = indices {
                self.faces.push(face);
            }
        }

        Ok(())
    }

    /// Add a vertex to the mesh, evaluating the surface at the given UV.
    fn add_surface_vertex(&mut self, uv: Vector2<T>) -> usize {
        let p = self.surface.point_at(uv.x, uv.y);
        let n = self.surface.normal_at(uv.x, uv.y);
        let idx = self.points.len();
        self.points.push(p);
        self.normals.push(n);
        self.uvs.push(uv);
        idx
    }

    /// Find an existing vertex by UV proximity, or add a new one.
    fn find_or_add_vertex(
        &mut self,
        point: Point3<T>,
        normal: Vector3<T>,
        uv: Vector2<T>,
    ) -> usize {
        let eps = T::from_f64(1e-10).unwrap();
        for (i, existing_uv) in self.uvs.iter().enumerate() {
            if (existing_uv - uv).norm() < eps {
                return i;
            }
        }
        let idx = self.points.len();
        self.points.push(point);
        self.normals.push(normal);
        self.uvs.push(uv);
        idx
    }
}
