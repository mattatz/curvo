use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OMatrix,
    OPoint, OVector, Point3, Point4, RealField, Vector2, Vector3, U1,
};
use simba::scalar::SupersetOf;

use crate::{
    curve::nurbs_curve::{dehomogenize, NurbsCurve, NurbsCurve3D},
    misc::binomial::Binomial,
    misc::transformable::Transformable,
    misc::FloatingPoint,
    misc::Ray,
    prelude::{KnotVector, SurfaceTessellation},
    tessellation::adaptive_tessellation_node::AdaptiveTessellationNode,
    tessellation::adaptive_tessellation_processor::{
        AdaptiveTessellationOptions, AdaptiveTessellationProcessor,
    },
    tessellation::surface_point::SurfacePoint,
};

/// NURBS surface representation
/// by generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug)]
pub struct NurbsSurface<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// control points with homogeneous coordinates
    /// the last element of the vector is the `weight`
    control_points: Vec<Vec<OPoint<T, D>>>,
    u_degree: usize,
    v_degree: usize,
    u_knots: KnotVector<T>,
    v_knots: KnotVector<T>,
}

/// 2D NURBS surface alias
pub type NurbsSurface2D<T> = NurbsSurface<T, Const<3>>;
/// 3D NURBS surface alias
pub type NurbsSurface3D<T> = NurbsSurface<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    pub fn new(
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<T>,
        v_knots: Vec<T>,
        control_points: Vec<Vec<OPoint<T, D>>>,
    ) -> Self {
        Self {
            u_degree,
            v_degree,
            u_knots: KnotVector::new(u_knots),
            v_knots: KnotVector::new(v_knots),
            control_points,
        }
    }

    /// Get the u domain of the knot vector by degree
    pub fn u_knots_domain(&self) -> (T, T) {
        self.u_knots.domain(self.u_degree)
    }

    /// Get the v domain of the knot vector by degree
    pub fn v_knots_domain(&self) -> (T, T) {
        self.v_knots.domain(self.v_degree)
    }

    /// Evaluate the surface at the given u, v parameters to get a point
    pub fn point_at(&self, u: T, v: T) -> OPoint<T, DimNameDiff<D, U1>> {
        let p = self.point(u, v);
        dehomogenize(&p).unwrap()
    }

    /// Evaluate the surface at the given u, v parameters to get a point
    pub fn point(&self, u: T, v: T) -> OPoint<T, D> {
        let n = self.u_knots.len() - self.u_degree - 2;
        let m = self.v_knots.len() - self.v_degree - 2;

        let knot_span_index_u = self.u_knots.find_knot_span_index(n, self.u_degree, u);
        let knot_span_index_v = self.v_knots.find_knot_span_index(m, self.v_degree, v);
        let u_basis_vals = self
            .u_knots
            .basis_functions(knot_span_index_u, u, self.u_degree);
        let v_basis_vals = self
            .v_knots
            .basis_functions(knot_span_index_v, v, self.v_degree);
        let uind = knot_span_index_u - self.u_degree;

        let mut position = OPoint::<T, D>::origin();
        for l in 0..=self.v_degree {
            let mut temp = OPoint::<T, D>::origin();
            let vind = knot_span_index_v - self.v_degree + l;

            // sample u isoline
            for k in 0..=self.u_degree {
                temp.coords += &self.control_points[uind + k][vind].coords * u_basis_vals[k];
            }

            // add point from u isoline
            position.coords += temp.coords * v_basis_vals[l];
        }

        position
    }

    // Compute a regularly spaced grid of points on surface.
    // Generally, this algorithm is faster than directly evaluating these by pre-computing all of the basis functions
    pub fn regular_sample_points(
        &self,
        divs_u: usize,
        divs_v: usize,
    ) -> Vec<Vec<OPoint<T, DimNameDiff<D, U1>>>> {
        let (knot_spans_u, bases_u) = self
            .u_knots
            .regulary_spaced_basis_functions(self.u_degree, divs_u);
        let (knot_spans_v, bases_v) = self
            .v_knots
            .regulary_spaced_basis_functions(self.v_degree, divs_v);
        let mut pts = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];

            for j in 0..=divs_v {
                let pt = self.point_given_bases_knot_spans(
                    knot_spans_u[i],
                    knot_spans_v[j],
                    &bases_u[i],
                    &bases_v[j],
                );
                row.push(dehomogenize(&pt).unwrap());
            }

            pts.push(row);
        }

        pts
    }

    /// Compute a point on the surface given the basis functions and knot spans
    fn point_given_bases_knot_spans(
        &self,
        knot_span_u: usize,
        knot_span_v: usize,
        bases_u: &[T],
        bases_v: &[T],
    ) -> OPoint<T, D> {
        let mut position = OPoint::<T, D>::origin();

        // could be precomputed
        let uind = knot_span_u - self.u_degree;
        let mut vind = knot_span_v - self.v_degree;

        for l in 0..(self.v_degree + 1) {
            let mut temp = OPoint::<T, D>::origin();

            for k in 0..(self.u_degree + 1) {
                temp.coords += &self.control_points[uind + k][vind].coords * bases_u[k];
            }

            vind += 1;

            position.coords += temp.coords * bases_v[l];
        }

        position
    }

    // Compute a regularly spaced grid of normals on surface.
    pub fn regular_sample_normals(
        &self,
        divs_u: usize,
        divs_v: usize,
    ) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> {
        let ders = self.regular_sample_rational_derivatives(divs_u, divs_v, 1);
        ders.into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|der| {
                        let v0 = &der[1][0];
                        let v1 = &der[0][1];
                        v0.cross(v1).normalize()
                    })
                    .collect()
            })
            .collect()
    }

    // Compute a regularly spaced grid of rational derivatives on surface.
    #[allow(clippy::type_complexity)]
    pub fn regular_sample_rational_derivatives(
        &self,
        divs_u: usize,
        divs_v: usize,
        derivs: usize,
    ) -> Vec<Vec<Vec<Vec<OVector<T, DimNameDiff<D, U1>>>>>> {
        let ders = self.regular_sample_derivatives(divs_u, divs_v, derivs);

        let mut rat_ders = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];
            for j in 0..=divs_v {
                let skl = rational_derivatives(&ders[i][j], derivs);
                row.push(skl);
            }
            rat_ders.push(row);
        }

        rat_ders
    }

    /// Compute a regularly spaced grid of derivatives on a surface.
    #[allow(clippy::type_complexity)]
    pub fn regular_sample_derivatives(
        &self,
        divs_u: usize,
        divs_v: usize,
        derivs: usize,
    ) -> Vec<Vec<Vec<Vec<OVector<T, D>>>>> {
        let (knot_spans_u, bases_u) = self
            .u_knots
            .regularly_spaced_derivative_basis_functions(self.u_degree, divs_u);
        let (knot_spans_v, bases_v) = self
            .v_knots
            .regularly_spaced_derivative_basis_functions(self.v_degree, divs_v);
        let mut ders = vec![];

        for i in 0..=divs_u {
            let mut row = vec![];

            for j in 0..=divs_v {
                row.push(self.derivatives_given_bases_knot_spans(
                    knot_spans_u[i],
                    knot_spans_v[j],
                    &bases_u[i],
                    &bases_v[j],
                    derivs,
                ));
            }

            ders.push(row);
        }

        ders
    }

    /// Compute the derivatives given the basis functions and knot spans
    pub fn derivatives_given_bases_knot_spans(
        &self,
        knot_span_u: usize,
        knot_span_v: usize,
        bases_u: &[Vec<T>],
        bases_v: &[Vec<T>],
        derivs: usize,
    ) -> Vec<Vec<OVector<T, D>>> {
        let du = if derivs < self.u_degree {
            derivs
        } else {
            self.u_degree
        };
        let dv = if derivs < self.v_degree {
            derivs
        } else {
            self.v_degree
        };
        let mut skl = vec![vec![OVector::<T, D>::zeros(); du + 1]; dv + 1];

        for k in 0..=du {
            let mut temp = vec![];
            for s in 0..=self.v_degree {
                let mut row = OVector::<T, D>::zeros();
                for r in 0..=self.u_degree {
                    row += &self.control_points[knot_span_u - self.u_degree + r]
                        [knot_span_v - self.v_degree + s]
                        .coords
                        * bases_u[k][r];
                }
                temp.push(row);
            }

            let nk = derivs - k;
            let dd = if nk < dv { nk } else { dv };

            for l in 0..=dd {
                let mut v = OVector::<T, D>::zeros();
                for s in 0..=self.v_degree {
                    v += &temp[s] * bases_v[l][s];
                }
                skl[k][l] = v;
            }
        }

        skl
    }

    /// Tessellate the surface into a meshable form
    /// if adaptive_options is None, the surface will be tessellated at control points
    /// or else it will be tessellated adaptively based on the options
    /// this `adaptive` means that the surface will be tessellated based on the curvature of the surface
    pub fn tessellate(
        &self,
        adaptive_options: Option<AdaptiveTessellationOptions<T>>,
    ) -> SurfaceTessellation<T, D> {
        let is_adaptive = adaptive_options.is_some();
        let mut options = adaptive_options.unwrap_or_default();

        let min_u = (self.control_points.len() - 1) * 2;
        let min_v = (self.control_points[0].len() - 1) * 2;

        options.min_divs_u = options.min_divs_u.max(min_u);
        options.min_divs_v = options.min_divs_v.max(min_v);

        let divs_u = options.min_divs_u;
        let divs_v = options.min_divs_v;

        let (umin, umax) = self.u_knots_domain();
        let (vmin, vmax) = self.v_knots_domain();
        let du = (umax - umin) / T::from_usize(divs_u).unwrap();
        let dv = (vmax - vmin) / T::from_usize(divs_v).unwrap();

        let mut pts = vec![];

        for i in 0..=divs_v {
            let mut row = vec![];
            for j in 0..=divs_u {
                let u = umin + du * T::from_usize(j).unwrap();
                let v = vmin + dv * T::from_usize(i).unwrap();

                let ds = self.rational_derivatives(u, v, 1);
                let norm = ds[0][1].cross(&ds[1][0]).normalize();
                row.push(SurfacePoint {
                    point: ds[0][0].clone().into(),
                    normal: norm,
                    uv: Vector2::new(u, v),
                    is_normal_degenerated: false,
                });
            }
            pts.push(row);
        }

        let mut divs = vec![];
        for i in 0..divs_v {
            for j in 0..divs_u {
                let iv = divs_v - i;
                let corners = [
                    pts[iv - 1][j].clone(),
                    pts[iv - 1][j + 1].clone(),
                    pts[iv][j + 1].clone(),
                    pts[iv][j].clone(),
                ];
                let node = AdaptiveTessellationNode::new(divs.len(), corners, None);
                divs.push(node);
            }
        }

        let nodes = if !is_adaptive {
            divs
        } else {
            let mut processor = AdaptiveTessellationProcessor {
                surface: self,
                nodes: divs,
            };

            for i in 0..divs_v {
                for j in 0..divs_u {
                    let ci = i * divs_u + j;
                    let s = processor.south(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let e = processor.east(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let n = processor.north(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let w = processor.west(ci, i, j, divs_u, divs_v).map(|n| n.id);
                    let node = processor.nodes.get_mut(ci).unwrap();
                    node.neighbors = [s, e, n, w];
                    processor.divide(ci, &options);
                }
            }

            processor.nodes
        };

        SurfaceTessellation::new(self, &nodes)
    }

    /// Regularly tessellate the surface into a meshable form
    /// This tessellation is faster than adaptive one because of pre-computed basis functions
    /// There is trade-off between speed and mesh quality
    pub fn regular_tessellate(&self, divs_u: usize, divs_v: usize) -> SurfaceTessellation<T, D> {
        let points = self.regular_sample_points(divs_u, divs_v);
        let ders = self.regular_sample_normals(divs_u, divs_v);
        let u_span = self.u_knots.regularly_spaced_span(self.u_degree, divs_u);
        let v_span = self.v_knots.regularly_spaced_span(self.v_degree, divs_v);

        let points: Vec<_> = points.into_iter().flatten().collect();
        let normals: Vec<_> = ders.into_iter().flatten().collect();
        let faces = (0..divs_u)
            .flat_map(|iu| {
                let ioff = iu * (divs_v + 1);
                (0..divs_v).flat_map(move |iv| {
                    [
                        [ioff + iv, ioff + iv + 1, ioff + iv + divs_v + 2],
                        [ioff + iv, ioff + iv + divs_v + 2, ioff + iv + divs_v + 1],
                    ]
                })
            })
            .collect();
        let uvs = (0..=divs_u)
            .flat_map(|iu| {
                let iu = T::from_usize(iu).unwrap();
                let u = u_span.0 + u_span.2 * iu;
                (0..=divs_v).map(move |iv| {
                    let iv = T::from_usize(iv).unwrap();
                    let v = v_span.0 + v_span.2 * iv;
                    Vector2::new(u, v)
                })
            })
            .collect();

        SurfaceTessellation {
            points,
            normals,
            faces,
            uvs,
        }
    }

    /// Evaluate the normal at the given u, v parameters
    pub fn normal_at(&self, u: T, v: T) -> OVector<T, DimNameDiff<D, U1>> {
        let deriv = self.rational_derivatives(u, v, 1);
        let v0 = &deriv[1][0];
        let v1 = &deriv[0][1];
        v0.cross(v1)
    }

    /// Evaluate the rational derivatives at the given u, v parameters
    pub fn rational_derivatives(
        &self,
        u: T,
        v: T,
        derivs: usize,
    ) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> {
        let ders = self.derivatives(u, v, derivs);
        rational_derivatives(&ders, derivs)
    }

    /// Evaluate the derivatives at the given u, v parameters
    fn derivatives(&self, u: T, v: T, derivs: usize) -> Vec<Vec<OVector<T, D>>> {
        let n = self.u_knots.len() - self.u_degree - 2;
        let m = self.v_knots.len() - self.v_degree - 2;

        let du = if derivs < self.u_degree {
            derivs
        } else {
            self.u_degree
        };
        let dv = if derivs < self.v_degree {
            derivs
        } else {
            self.v_degree
        };
        let mut skl = vec![vec![OVector::<T, D>::zeros(); derivs + 1]; derivs + 1];
        let knot_span_index_u = self.u_knots.find_knot_span_index(n, self.u_degree, u);
        let knot_span_index_v = self.v_knots.find_knot_span_index(m, self.v_degree, v);
        let uders = self
            .u_knots
            .derivative_basis_functions(knot_span_index_u, u, self.u_degree, n);
        let vders = self
            .v_knots
            .derivative_basis_functions(knot_span_index_v, v, self.v_degree, m);
        let mut temp = vec![OPoint::<T, D>::origin(); self.v_degree + 1];

        for k in 0..=du {
            for s in 0..=self.v_degree {
                temp[s] = OPoint::<T, D>::origin();
                for r in 0..=self.u_degree {
                    let w = &self.control_points[knot_span_index_u - self.u_degree + r]
                        [knot_span_index_v - self.v_degree + s]
                        * uders[k][r];
                    let column = temp.get_mut(s).unwrap();
                    w.coords.iter().enumerate().for_each(|(i, v)| {
                        column[i] += *v;
                    });
                }
            }

            let nk = derivs - k;
            let dd = if nk < dv { nk } else { dv };

            for l in 0..=dd {
                for (s, item) in temp.iter().enumerate().take(self.v_degree + 1) {
                    let w = item * vders[l][s];
                    let column = skl[k].get_mut(l).unwrap();
                    w.coords.iter().enumerate().for_each(|(i, v)| {
                        column[i] += *v;
                    });
                }
            }
        }

        skl
    }

    /// Extrude a NURBS curve to create a NURBS surface
    pub fn extrude(profile: &NurbsCurve<T, D>, axis: OVector<T, DimNameDiff<D, U1>>) -> Self {
        let prof_points = profile.dehomogenized_control_points();
        let prof_weights = profile.weights();

        let translation = axis.clone();
        let half_translation = &translation * T::from_f64(0.5).unwrap();

        let mut control_points = vec![vec![], vec![], vec![]];

        for i in 0..prof_points.len() {
            let p0 = &prof_points[i] + &translation;
            let p1 = &prof_points[i] + &half_translation;
            let p2 = &prof_points[i];

            let mut o0 = OPoint::<T, D>::origin();
            let mut o1 = OPoint::<T, D>::origin();
            let mut o2 = OPoint::<T, D>::origin();
            let w = prof_weights[i];
            for j in 0..D::dim() - 1 {
                o0[j] = p0[j] * w;
                o1[j] = p1[j] * w;
                o2[j] = p2[j] * w;
            }
            o0[D::dim() - 1] = w;
            o1[D::dim() - 1] = w;
            o2[D::dim() - 1] = w;

            control_points[0].push(o0);
            control_points[1].push(o1);
            control_points[2].push(o2);
        }

        // dbg!(profile.control_points());
        // dbg!(&control_points);

        Self {
            u_degree: 2,
            v_degree: profile.degree(),
            u_knots: KnotVector::new(vec![
                T::zero(),
                T::zero(),
                T::zero(),
                T::one(),
                T::one(),
                T::one(),
            ]),
            v_knots: profile.knots().clone(),
            control_points,
        }
    }

    /// Try to loft a collection of curves to create a surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Translation3};
    ///
    /// // Create a collection of curves
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 2.0, 0.),
    ///     Point3::new(1.0, 2.5, 0.),
    /// ];
    /// let interpolated = NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    /// // Offset the curve by translation of 2.0 in the z direction
    /// let offsetted = interpolated.transformed(&Translation3::new(0.0, 0.0, 2.0).into());
    ///
    /// // Loft the curves to create a NURBS surface
    /// let lofted = NurbsSurface::try_loft(&[interpolated, offsetted], None);
    /// assert!(lofted.is_ok());
    /// ```
    pub fn try_loft(curves: &[NurbsCurve<T, D>], degree_v: Option<usize>) -> anyhow::Result<Self> {
        let unified_curves = try_unify_curve_knot_vectors(curves)?;

        let degree_u = unified_curves[0].degree();
        let degree_v = degree_v.unwrap_or(degree_u).min(unified_curves.len() - 1);

        let knots_u = unified_curves[0].knots().clone();
        let mut control_points = vec![];

        // Interpolate each column of control points to get the nurbs curve aligned with the v direction
        let v_curves: anyhow::Result<Vec<_>> = (0..unified_curves[0].control_points().len())
            .map(|i| {
                let points = unified_curves
                    .iter()
                    .map(|c| dehomogenize(&c.control_points()[i]).unwrap())
                    .collect::<Vec<_>>();
                NurbsCurve::try_interpolate(&points, degree_v)
            })
            .collect();
        let v_curves = v_curves?;
        v_curves.iter().for_each(|c| {
            control_points.push(c.control_points().clone());
        });
        let knots_v = v_curves.last().unwrap().knots().clone();

        Ok(Self {
            control_points,
            u_degree: degree_u,
            v_degree: degree_v,
            u_knots: knots_u,
            v_knots: knots_v,
        })
    }

    /// Cast the surface to a surface with another floating point type
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> NurbsSurface<F, D>
    where
        DefaultAllocator: Allocator<F, D>,
    {
        NurbsSurface {
            control_points: self
                .control_points
                .iter()
                .map(|row| row.iter().map(|p| p.clone().cast()).collect())
                .collect(),
            u_degree: self.u_degree,
            v_degree: self.v_degree,
            u_knots: self.u_knots.cast(),
            v_knots: self.v_knots.cast(),
        }
    }
}

/// Compute the rational derivatives of derivatives
fn rational_derivatives<T, D>(
    ders: &[Vec<OVector<T, D>>],
    derivs: usize,
) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>>
where
    T: FloatingPoint,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    let a_ders: Vec<_> = ders
        .iter()
        .map(|row| {
            row.iter()
                .map(|d| {
                    let mut a_ders = vec![];
                    for i in 0..D::dim() - 1 {
                        a_ders.push(d[i]);
                    }
                    OVector::<T, DimNameDiff<D, U1>>::from_vec(a_ders)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let w_ders: Vec<_> = ders
        .iter()
        .map(|row| row.iter().map(|d| d[D::dim() - 1]).collect::<Vec<_>>())
        .collect();

    let mut skl: Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> = vec![];
    let mut binom = Binomial::<T>::new();

    for k in 0..=derivs {
        let mut row = vec![];

        for l in 0..=(derivs - k) {
            let mut v = a_ders[k][l].clone();
            for j in 1..=l {
                let coef = binom.get(l, j) * w_ders[0][j];
                v -= &row[l - j] * coef;
            }

            for i in 1..=k {
                let coef = binom.get(k, i) * w_ders[i][0];
                v -= &skl[k - i][l] * coef;
                let mut v2 = OVector::<T, DimNameDiff<D, U1>>::zeros();
                for j in 1..=l {
                    v2 += &skl[k - i][l - j] * binom.get(l, j) * w_ders[i][j];
                }
                v -= v2 * binom.get(k, i);
            }

            let v = v / w_ders[0][0];
            row.push(v);
        }

        skl.push(row);
    }

    skl
}

/// A specialized trait for NURBS surfaces with 3D points,
/// particularly designed for sweeping operations that require Frenet frames.
impl<T: FloatingPoint> NurbsSurface3D<T> {
    /// Try to sweep a profile curve along a rail curve to create a surface
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Translation3};
    ///
    /// // Create a collection of curves
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let profile = NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    ///
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(1.0, 0., 0.),
    ///     Point3::new(0.0, -1., 1.),
    ///     Point3::new(-1.0, 0., 2.),
    ///     Point3::new(0.0, 1., 3.),
    /// ];
    /// let rail= NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    ///
    /// // Sweep the profile curve along the rail curve to create a NURBS surface
    /// let swept = NurbsSurface::try_sweep(&profile, &rail, Some(3));
    /// assert!(swept.is_ok());
    /// ```
    pub fn try_sweep(
        profile: &NurbsCurve3D<T>,
        rail: &NurbsCurve3D<T>,
        degree_v: Option<usize>,
    ) -> anyhow::Result<Self> {
        let (start, end) = rail.knots_domain();
        let _pt0 = rail.point_at(start);
        let samples = rail.control_points().len() * 2;
        let span = (end - start) / T::from_usize(samples - 1).unwrap();

        let parameters: Vec<_> = (0..samples)
            .map(|i| start + T::from_usize(i).unwrap() * span)
            .collect();

        let frames = rail.compute_frenet_frames(&parameters);
        let curves: Vec<_> = frames
            .iter()
            .map(|frame| {
                let transform = frame.matrix();
                profile.transformed(&transform.into())
            })
            .collect();

        Self::try_loft(&curves, degree_v)
    }

    /// Try to revolve a profile curve around an axis to create a surface
    /// /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::{Point3, Vector3};
    ///
    /// // Create a profile curve to revolve
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let profile = NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    ///
    /// // Revolve the profile curve around the z-axis by PI radians to create a NURBS surface
    /// let reolved = NurbsSurface::try_revolve(&profile, &Point3::origin(), &Vector3::z_axis(), std::f64::consts::PI);
    /// assert!(reolved.is_ok());
    /// ```
    pub fn try_revolve(
        profile: &NurbsCurve3D<T>,
        center: &Point3<T>,
        axis: &Vector3<T>,
        theta: T,
    ) -> anyhow::Result<Self> {
        let prof_points = profile.dehomogenized_control_points();
        let prof_weights = profile.weights();

        let two = T::from_f64(2.0).unwrap();
        let (narcs, mut u_knots) = if theta <= T::pi() / two {
            (1, vec![T::zero(); 6])
        } else if theta <= T::pi() {
            let mut knots = vec![T::zero(); 6 + 2];
            let half = T::from_f64(0.5).unwrap();
            knots[3] = half;
            knots[4] = half;
            (2, knots)
        } else if theta <= T::from_f64(3.0).unwrap() * T::pi() / two {
            let mut knots = vec![T::zero(); 6 + 2 * 2];
            let frac_three = T::from_f64(1.0 / 3.0).unwrap();
            let two_frac_three = T::from_f64(2.0 / 3.0).unwrap();
            knots[3] = frac_three;
            knots[4] = frac_three;
            knots[5] = two_frac_three;
            knots[6] = two_frac_three;
            (3, knots)
        } else {
            let mut knots = vec![T::zero(); 6 + 2 * 3];
            let frac_four = T::from_f64(1.0 / 4.0).unwrap();
            let half = T::from_f64(0.5).unwrap();
            let three_frac_four = T::from_f64(3.0 / 4.0).unwrap();
            knots[3] = frac_four;
            knots[4] = frac_four;
            knots[5] = half;
            knots[6] = half;
            knots[7] = three_frac_four;
            knots[8] = three_frac_four;
            (4, knots)
        };

        let dtheta = theta / T::from_usize(narcs).unwrap();
        let j = 3 + 2 * (narcs - 1);

        for i in 0..3 {
            u_knots[i] = T::zero();
            u_knots[j + i] = T::one();
        }

        let wm = (dtheta / two).cos();

        let mut angle = T::zero();
        let mut sines = vec![T::zero(); narcs + 1];
        let mut cosines = vec![T::zero(); narcs + 1];
        for i in 0..=narcs {
            cosines[i] = angle.cos();
            sines[i] = angle.sin();
            angle += dtheta;
        }

        let mut control_points = vec![vec![Point4::origin(); prof_points.len()]; 2 * narcs + 1];

        for j in 0..prof_points.len() {
            let p = &prof_points[j];
            let s = (p - center).dot(axis);
            let o = center + axis * s;

            // vector from the axis
            let mut x = p - o;
            // radius at height
            let r = x.norm();
            // perpendicular vector to x & axis
            let mut y = axis.cross(&x);

            if r > T::default_epsilon() {
                x *= T::one() / r;
                y *= T::one() / r;
            }

            // control_points[0][j]
            let mut p0 = prof_points[j];
            control_points[0][j].x = p0.x;
            control_points[0][j].y = p0.y;
            control_points[0][j].z = p0.z;
            control_points[0][j].w = prof_weights[j];

            let mut t0 = y;
            let mut index = 0;
            for i in 1..=narcs {
                let p2 = if r <= T::default_epsilon() {
                    o
                } else {
                    o + x * cosines[i] * r + y * sines[i] * r
                };

                let k = index + 2;
                control_points[k][j].x = p2.x;
                control_points[k][j].y = p2.y;
                control_points[k][j].z = p2.z;
                control_points[k][j].w = prof_weights[j];

                let t2 = y * cosines[i] - x * sines[i];

                let l = index + 1;
                if r <= T::default_epsilon() {
                    control_points[l][j].x = o.x;
                    control_points[l][j].y = o.y;
                    control_points[l][j].z = o.z;
                } else {
                    let nt0 = t0.normalize();
                    let nt2 = t2.normalize();
                    let r0 = Ray::new(p0, nt0);
                    let r1 = Ray::new(p2, nt2);
                    let intersection = r0
                        .find_intersection(&r1)
                        .ok_or(anyhow::anyhow!("No intersection between rays"))?;

                    let intersected = intersection.intersection0.0;
                    control_points[l][j].x = intersected.x;
                    control_points[l][j].y = intersected.y;
                    control_points[l][j].z = intersected.z;
                }

                control_points[l][j].w = wm * prof_weights[j];

                index += 2;

                if i < narcs {
                    p0 = p2;
                    t0 = t2;
                }
            }
        }

        // Scale the control points by their weights to make them rational
        control_points.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|p| {
                let w = p.w;
                p.x *= w;
                p.y *= w;
                p.z *= w;
            });
        });

        Ok(Self {
            control_points,
            u_degree: 2,
            v_degree: profile.degree(),
            u_knots: KnotVector::new(u_knots),
            v_knots: profile.knots().clone(),
        })
    }
}

/// Unify the knot vectors of a collection of NURBS curves
///
fn try_unify_curve_knot_vectors<T, D>(
    curves: &[NurbsCurve<T, D>],
) -> anyhow::Result<Vec<NurbsCurve<T, D>>>
where
    T: FloatingPoint,
    D: DimName,
    DefaultAllocator: Allocator<T, D>,
{
    let max_degree = curves.iter().fold(0, |d, c| d.max(c.degree()));

    // elevate all curves to the same degree
    let mut curves = curves
        .iter()
        .map(|c| {
            if c.degree() < max_degree {
                c.try_elevate_degree(max_degree)
            } else {
                Ok(c.clone())
            }
        })
        .collect::<anyhow::Result<Vec<NurbsCurve<T, D>>>>()?;

    let knot_intervals = curves
        .iter()
        .map(|c| {
            let knots = c.knots();
            let min = knots[0];
            let max = knots[knots.len() - 1];
            (min, max)
        })
        .collect::<Vec<_>>();

    //shift all knot vectors to start at 0.0
    curves.iter_mut().enumerate().for_each(|(i, c)| {
        let min = knot_intervals[i].0;
        c.knots_mut().iter_mut().for_each(|x| *x -= min);
    });

    //find the max knot span
    let knot_spans = knot_intervals
        .iter()
        .map(|(min, max)| *max - *min)
        .collect::<Vec<_>>();
    let max_knot_span = knot_spans.iter().fold(T::zero(), |x, a| a.max(x));

    //scale all of the knot vectors to match
    curves.iter_mut().enumerate().for_each(|(i, c)| {
        let scale = max_knot_span / knot_spans[i];
        c.knots_mut().iter_mut().for_each(|x| *x *= scale);
    });

    //merge all of the knot vectors
    let merged_knots = curves
        .iter()
        .fold(vec![], |a, c| sorted_set_union(c.knots().as_slice(), &a));

    //knot refinement on each curve
    for curve in curves.iter_mut() {
        let rem = sorted_set_sub(&merged_knots, curve.knots().as_slice());
        if !rem.is_empty() {
            curve.try_refine_knot(rem)?;
        }
    }

    Ok(curves)
}

fn sorted_set_union<T: RealField + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut merged = Vec::new();
    let mut ai = 0;
    let mut bi = 0;
    while ai < a.len() || bi < b.len() {
        if ai >= a.len() {
            merged.push(b[bi]);
            bi += 1;
            continue;
        } else if bi >= b.len() {
            merged.push(a[ai]);
            ai += 1;
            continue;
        }

        let diff = a[ai] - b[bi];

        if diff.abs() < T::default_epsilon() {
            merged.push(a[ai]);
            ai += 1;
            bi += 1;
            continue;
        }

        if diff > T::zero() {
            // add the smallar value
            merged.push(b[bi]);
            bi += 1;
            continue;
        }

        // thus diff < 0.0
        merged.push(a[ai]);
        ai += 1;
    }

    merged
}

fn sorted_set_sub<T: RealField + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut ai = 0;
    let mut bi = 0;

    while ai < a.len() {
        if bi >= b.len() {
            result.push(a[ai]);
            ai += 1;
            continue;
        }

        if (a[ai] - b[bi]).abs() < T::default_epsilon() {
            ai += 1;
            bi += 1;
            continue;
        }

        result.push(a[ai]);
        ai += 1;
    }

    result
}

/// Enable to transform a NURBS surface by a given DxD matrix
impl<'a, T: FloatingPoint, const D: usize> Transformable<&'a OMatrix<T, Const<D>, Const<D>>>
    for NurbsSurface<T, Const<D>>
{
    fn transform(&mut self, transform: &'a OMatrix<T, Const<D>, Const<D>>) {
        self.control_points.iter_mut().for_each(|rows| {
            rows.iter_mut().for_each(|p| {
                // dehomogenize
                let ow = p[D - 1];
                let mut pt = *p;
                for i in 0..D - 1 {
                    pt[i] /= ow;
                }

                pt[D - 1] = T::one();
                let transformed = transform * pt;
                let w = transformed[D - 1];
                for i in 0..D - 1 {
                    p[i] = transformed[i] / w * ow;
                }
            });
        });
    }
}

/*
impl<T: FloatingPoint + SubsetOf<F>, F: FloatingPoint, D: DimName> Castible<NurbsSurface<F, D>>
    for NurbsSurface<T, D>
where
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<F, D>,
{
    fn cast(&self) -> NurbsSurface<F, D> {
        NurbsSurface {
            control_points: self
                .control_points
                .iter()
                .map(|row| row.iter().map(|p| p.clone().cast()).collect())
                .collect(),
            u_degree: self.u_degree,
            v_degree: self.v_degree,
            u_knots: self.u_knots.cast(),
            v_knots: self.v_knots.cast(),
        }
    }
}
*/
