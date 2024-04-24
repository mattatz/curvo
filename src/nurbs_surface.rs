use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, Matrix3,
    Matrix4, OPoint, OVector, Point2, Point3, RealField, Vector2, U1,
};

use crate::{
    adaptive_tessellation_node::AdaptiveTessellationNode,
    adaptive_tessellation_processor::{AdaptiveTessellationOptions, AdaptiveTessellationProcessor},
    binomial::Binomial,
    nurbs_curve::{dehomogenize, NurbsCurve},
    prelude::{KnotVector, SurfaceTessellation},
    transformable::Transformable,
    FloatingPoint, SurfacePoint,
};

/// NURBS surface representation
/// by generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug)]
pub struct NurbsSurface<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
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
    /// use approx::assert_relative_eq;
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
    /// let interpolated = NurbsCurve3D::try_interpolate(&points, 3, None, None).unwrap();
    /// // Offset the curve by translation of 2.0 in the z direction
    /// let offsetted = interpolated.transformed(&Translation3::new(0.0, 0.0, 2.0).into());
    ///
    /// // Loft the curves to create a NURBS surface
    /// let lofted = NurbsSurface::try_loft(&[interpolated, offsetted], None);
    /// assert!(lofted.is_ok());
    /// ```
    pub fn try_loft(curves: &[NurbsCurve<T, D>], degree_v: Option<usize>) -> anyhow::Result<Self> {
        let unified_curves = unify_curve_knot_vectors(curves)?;

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
                NurbsCurve::try_interpolate(&points, degree_v, None, None)
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
}

/// Unify the knot vectors of a collection of NURBS curves
///
fn unify_curve_knot_vectors<T, D>(
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
            curve.knot_refine(rem);
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

/// Enable to transform a NURBS surface by a given 3x3 matrix
impl<T: FloatingPoint> Transformable<&Matrix3<T>> for NurbsSurface2D<T> {
    fn transform(&mut self, transform: &Matrix3<T>) {
        self.control_points.iter_mut().for_each(|rows| {
            rows.iter_mut().for_each(|p| {
                let hom = Point2::new(p.x, p.y).to_homogeneous();
                let transformed = transform * hom;
                p.x = transformed.x;
                p.y = transformed.y;
            });
        });
    }
}

/// Enable to transform a NURBS surface by a given 4x4 matrix
impl<T: FloatingPoint> Transformable<&Matrix4<T>> for NurbsSurface3D<T> {
    fn transform(&mut self, transform: &Matrix4<T>) {
        self.control_points.iter_mut().for_each(|rows| {
            rows.iter_mut().for_each(|p| {
                let hom = Point3::new(p.x, p.y, p.z).to_homogeneous();
                let transformed = transform * hom;
                p.x = transformed.x;
                p.y = transformed.y;
                p.z = transformed.z;
            });
        });
    }
}
