use nalgebra::allocator::Allocator;
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff, DimNameSub,
    Matrix3, Matrix4, OPoint, OVector, Point2, Point3, U1,
};
use rand::rngs::ThreadRng;
use rand::Rng;

use crate::binomial::Binomial;
use crate::prelude::{Invertible, KnotVector};
use crate::transformable::Transformable;
use crate::trigonometry::three_points_are_flat;
use crate::FloatingPoint;

/// NURBS curve representation
/// By generics, it can be used for 2D or 3D curves with f32 or f64 scalar types
#[derive(Clone, Debug)]
pub struct NurbsCurve<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<T, D>,
{
    control_points: Vec<OPoint<T, D>>,
    degree: usize,
    knots: KnotVector<T>,
}

/// 2D NURBS curve aliases
pub type NurbsCurve2D<T> = NurbsCurve<T, Const<3>>;
/// 3D NURBS curve aliases
pub type NurbsCurve3D<T> = NurbsCurve<T, Const<4>>;

impl<T: FloatingPoint, D: DimName> NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Create a new NURBS curve
    /// # Failures
    /// - if the number of control points is less than the degree
    /// - the number of knots is not equal to the number of control points + the degree + 1
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    ///
    /// let control_points: Vec<Point2<f64>> = vec![
    ///     Point2::new(50., 50.),
    ///     Point2::new(30., 370.),
    ///     Point2::new(180., 350.),
    ///     Point2::new(150., 100.),
    ///     Point2::new(250., 50.),
    ///     Point2::new(350., 100.),
    ///     Point2::new(470., 400.),
    /// ];
    /// let degree = 3;
    /// let m = control_points.len() + degree + 1;
    /// // create an uniform knot vector
    /// let knots = (0..m).map(|i| i as f64).collect();
    /// let nurbs = NurbsCurve::try_new(3, control_points, knots);
    /// assert!(nurbs.is_ok());
    /// ```
    pub fn try_new(
        degree: usize,
        control_points: Vec<OPoint<T, D>>,
        knots: Vec<T>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            control_points.len() > degree,
            "Too few control points for curve"
        );
        anyhow::ensure!(
            knots.len() == control_points.len() + degree + 1,
            "Invalid number of knots, got {}, expected {}",
            knots.len(),
            control_points.len() + degree + 1
        );

        let mut knots = knots.clone();
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(Self {
            degree,
            control_points,
            knots: KnotVector::new(knots),
        })
    }

    /// Create a dehomogenized version of the curve
    pub fn dehomogenize(&self) -> NurbsCurve<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        NurbsCurve {
            degree: self.degree,
            control_points: self.dehomogenized_control_points(),
            knots: self.knots.clone(),
        }
    }

    /// Return the dehomogenized control points
    pub fn dehomogenized_control_points(&self) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        self.control_points
            .iter()
            .map(|p| dehomogenize(p).unwrap())
            .collect()
    }

    pub fn weights(&self) -> Vec<T> {
        self.control_points
            .iter()
            .map(|p| p[D::dim() - 1])
            .collect()
    }

    /// Evaluate the curve at a given parameter to get a dehomonogenized point
    pub fn point_at(&self, t: T) -> OPoint<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let p = self.point(t);
        dehomogenize(&p).unwrap()
    }

    /// Sample the curve at a given number of points between the start and end parameters
    pub fn sample_regular_range(
        &self,
        start: T,
        end: T,
        samples: usize,
    ) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let mut points = vec![];
        let us = T::from_usize(samples).unwrap();
        let step = (end - start) / (us - T::one());
        for i in 0..samples {
            let t = start + T::from_usize(i).unwrap() * step;
            points.push(self.point_at(t));
        }
        points
    }

    /// Tessellate the curve using an adaptive algorithm
    /// this `adaptive` means that the curve will be tessellated based on the curvature of the curve
    pub fn tessellate(&self, tolerance: Option<T>) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        if self.degree == 1 {
            return self.dehomogenized_control_points();
        }

        let mut rng = rand::thread_rng();
        let tol = tolerance.unwrap_or(T::from_f64(1e-3).unwrap());
        let (start, end) = self.knots_domain();
        self.tessellate_adaptive(start, end, tol, &mut rng)
    }

    /// Tessellate the curve using an adaptive algorithm recursively
    /// if the curve between [start ~ end] is flat enough, it will return the two end points
    fn tessellate_adaptive(
        &self,
        start: T,
        end: T,
        tol: T,
        rng: &mut ThreadRng,
    ) -> Vec<OPoint<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let p1 = self.point_at(start);
        let p3 = self.point_at(end);

        let t = 0.5_f64 + 0.2_f64 * rng.gen::<f64>();
        let mid = start + (end - start) * T::from_f64(t).unwrap();
        let p2 = self.point_at(mid);

        let diff = &p1 - &p3;
        let diff2 = &p1 - &p2;

        if (diff.dot(&diff) < tol && diff2.dot(&diff2) > tol)
            || !three_points_are_flat(&p1, &p2, &p3, tol)
        {
            let exact_mid = start + (end - start) * T::from_f64(0.5).unwrap();
            let mut left_pts = self.tessellate_adaptive(start, exact_mid, tol, rng);
            let right_pts = self.tessellate_adaptive(exact_mid, end, tol, rng);
            left_pts.pop();
            [left_pts, right_pts].concat()
        } else {
            vec![p1, p3]
        }
    }

    /// Evaluate the curve at a given parameter to get a point
    fn point(&self, t: T) -> OPoint<T, D> {
        let n = self.knots.len() - self.degree - 2;
        let knot_span_index = self.knots.find_knot_span_index(n, self.degree, t);
        let basis = self.knots.basis_functions(knot_span_index, t, self.degree);
        let mut position = OPoint::<T, D>::origin();
        for i in 0..=self.degree {
            position.coords +=
                &self.control_points[knot_span_index - self.degree + i].coords * basis[i];
        }
        position
    }

    /// Evaluate the curve at a given parameter to get a tangent vector
    pub fn tangent_at(&self, u: T) -> OVector<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let deriv = self.rational_derivatives(u, 1);
        deriv[1].clone()
    }

    /// Evaluate the rational derivatives at a given parameter
    fn rational_derivatives(&self, u: T, derivs: usize) -> Vec<OVector<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let ders = self.derivatives(u, derivs);
        let a_ders: Vec<_> = ders
            .iter()
            .map(|d| {
                let mut a_ders = vec![];
                for i in 0..D::dim() - 1 {
                    a_ders.push(d[i]);
                }
                OVector::<T, DimNameDiff<D, U1>>::from_vec(a_ders)
            })
            .collect();
        let w_ders: Vec<_> = ders.iter().map(|d| d[D::dim() - 1]).collect();

        let mut ck = vec![];
        let mut binom = Binomial::<T>::new();
        for k in 0..=derivs {
            let mut v = a_ders[k].clone();

            for i in 1..=k {
                let coef = binom.get(k, i) * w_ders[i];
                v -= &ck[k - i] * coef;
            }

            let dehom = v / w_ders[0];
            ck.push(dehom);
        }
        ck
    }

    /// Evaluate the derivatives at a given parameter
    fn derivatives(&self, u: T, derivs: usize) -> Vec<OVector<T, D>> {
        let n = self.knots.len() - self.degree - 2;

        let du = if derivs < self.degree {
            derivs
        } else {
            self.degree
        };
        let mut derivatives = vec![OVector::<T, D>::zeros(); derivs + 1];

        let knot_span_index = self.knots.find_knot_span_index(n, self.degree, u);
        let nders = self
            .knots
            .derivative_basis_functions(knot_span_index, u, self.degree, du);
        for k in 0..=du {
            for j in 0..=self.degree {
                let w = &self.control_points[knot_span_index - self.degree + j] * nders[k][j];
                let column = derivatives.get_mut(k).unwrap();
                w.coords.iter().enumerate().for_each(|(i, v)| {
                    column[i] += *v;
                });
            }
        }

        derivatives
    }

    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn knots(&self) -> &KnotVector<T> {
        &self.knots
    }

    pub fn knots_mut(&mut self) -> &mut KnotVector<T> {
        &mut self.knots
    }

    pub fn control_points(&self) -> &Vec<OPoint<T, D>> {
        &self.control_points
    }

    pub fn knots_domain(&self) -> (T, T) {
        self.knots.domain(self.degree)
    }

    /// Try to create a periodic NURBS curve from a set of points
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    /// ];
    /// let closed = NurbsCurve3D::try_periodic(&points, 2);
    /// assert!(closed.is_ok());
    /// let curve = closed.unwrap();
    ///
    /// let (start, end) = curve.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = curve.point_at(start);
    /// let tail = curve.point_at(end);
    /// assert_relative_eq!(head, tail);
    /// ```
    pub fn try_periodic(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
    ) -> anyhow::Result<Self>
    where
        D: DimNameSub<U1>,
        <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
        DefaultAllocator: Allocator<T, D>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
        DefaultAllocator: Allocator<T, <<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
    {
        let n = points.len();
        if n < degree + 1 {
            anyhow::bail!("Too few control points for curve");
        }

        let pts: Vec<_> = (0..(n + degree))
            .map(|i| {
                if i < n {
                    points[i].clone()
                } else {
                    points[i - n].clone()
                }
            })
            .collect();

        let knots = (0..(n + 1 + degree * 2)).map(|i| T::from_usize(i).unwrap());

        Ok(Self {
            control_points: pts
                .iter()
                .map(|p| {
                    let coords = p.to_homogeneous();
                    OPoint::from_slice(coords.as_slice())
                })
                .collect(),
            degree,
            knots: KnotVector::from_iter(knots),
        })
    }

    /// Try to create an interpolated NURBS curve from a set of points
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 2.0, 0.),
    ///     Point3::new(1.0, 2.5, 0.),
    /// ];
    /// let interpolated = NurbsCurve3D::try_interpolate(&points, 3, None, None);
    /// assert!(interpolated.is_ok());
    /// let curve = interpolated.unwrap();
    ///
    /// let (start, end) = curve.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = curve.point_at(start);
    /// assert_relative_eq!(points[0], head);
    ///
    /// let tail = curve.point_at(end);
    /// assert_relative_eq!(points[points.len() - 1], tail);
    /// ```
    pub fn try_interpolate(
        points: &[OPoint<T, DimNameDiff<D, U1>>],
        degree: usize,
        start_tangent: Option<OVector<T, DimNameDiff<D, U1>>>,
        end_tangent: Option<OVector<T, DimNameDiff<D, U1>>>,
    ) -> anyhow::Result<Self>
    where
        DefaultAllocator: Allocator<T, D>,
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
    {
        let n = points.len();
        if n < degree + 1 {
            anyhow::bail!("Too few control points for curve");
        }

        let mut us: Vec<T> = vec![T::zero()];
        for i in 1..n {
            let sub = &points[i] - &points[i - 1];
            let chord = sub.norm();
            let last = us[i - 1];
            us.push(last + chord);
        }

        // normalize
        let max = us[us.len() - 1];
        for i in 0..us.len() {
            us[i] /= max;
        }

        let mut knots_start = vec![T::zero(); degree + 1];

        let has_tangent = start_tangent.is_some() && end_tangent.is_some();
        let (start, end) = if has_tangent {
            (0, us.len() - degree + 1)
        } else {
            (1, us.len() - degree)
        };

        for i in start..end {
            let mut weight_sums = T::zero();
            for j in 0..degree {
                weight_sums += us[i + j];
            }
            knots_start.push(weight_sums / T::from_usize(degree).unwrap());
        }

        let knots = KnotVector::new([knots_start, vec![T::one(); degree + 1]].concat());
        let plen = points.len();

        let (n, ld) = if has_tangent {
            (plen + 1, plen - (degree - 1))
        } else {
            (plen - 1, plen - (degree + 1))
        };

        // build basis function coefficients matrix

        let mut m_a = DMatrix::<T>::zeros(us.len(), degree + 1 + ld);
        for i in 0..us.len() {
            let u = us[i];
            let knot_span_index = knots.find_knot_span_index(n, degree, u);
            let basis = knots.basis_functions(knot_span_index, u, degree);

            let ls = knot_span_index - degree;
            let row_start = vec![T::zero(); ls];
            let row_end = vec![T::zero(); ld - ls];
            let e = [row_start, basis, row_end].concat();
            for j in 0..e.len() {
                m_a[(i, j)] = e[j];
            }
        }

        // dbg!(&mA);

        if has_tangent {
            let cols = m_a.ncols();
            let ln = cols - 2;
            let tan_row0 = [vec![-T::one(), T::one()], vec![T::zero(); ln]].concat();
            let tan_row1 = [vec![T::zero(); ln], vec![-T::one(), T::one()]].concat();
            // dbg!(&tan_row0);
            // dbg!(&tan_row1);
            m_a = m_a.insert_row(1, T::zero());
            let rows = m_a.nrows();
            m_a = m_a.insert_row(rows - 1, T::zero());
            for i in 0..cols {
                m_a[(1, i)] = tan_row0[i];
                m_a[(rows - 1, i)] = tan_row1[i];
            }
        }

        let dim = D::dim() - 1;

        let mult0 = knots[degree + 1] / T::from_usize(degree).unwrap();
        let mult1 = (T::one() - knots[knots.len() - degree - 2]) / T::from_usize(degree).unwrap();

        // solve Ax = b with LU decomposition
        let lu = m_a.lu();
        let mut m_x = DMatrix::<T>::identity(if has_tangent { plen + 2 } else { plen }, dim);
        let rows = m_x.nrows();
        for i in 0..dim {
            let b: Vec<_> = if has_tangent {
                let st = start_tangent.as_ref().unwrap()[i];
                let et = end_tangent.as_ref().unwrap()[i];
                let mut b = vec![points[0].coords[i]];
                b.push(mult0 * st);
                for j in 1..(plen - 1) {
                    b.push(points[j].coords[i]);
                }
                b.push(mult1 * et);
                b.push(points[plen - 1].coords[i]);
                b
            } else {
                points.iter().map(|p| p.coords[i]).collect()
            };

            let b = DVector::from_vec(b);
            // dbg!(&b);
            let xs = lu.solve(&b).ok_or(anyhow::anyhow!("Solve failed"))?;
            for j in 0..rows {
                m_x[(j, i)] = xs[j];
            }
        }

        // dbg!(&mX.shape());

        // extract control points from solved x
        let mut control_points = vec![];
        for i in 0..m_x.nrows() {
            let mut coords = vec![];
            for j in 0..m_x.ncols() {
                coords.push(m_x[(i, j)]);
            }
            coords.push(T::one());
            control_points.push(OPoint::from_slice(&coords));
        }

        // dbg!(control_points.len());
        // dbg!(knots.len());

        Ok(Self {
            degree,
            control_points,
            knots,
        })
    }

    /// Try to elevate the degree of the curve
    pub fn try_elevate_degree(&self, target_degree: usize) -> anyhow::Result<Self> {
        if target_degree <= self.degree {
            return Ok(self.clone());
        }

        let n = self.knots.len() - self.degree - 2;
        let new_degree = self.degree;
        let knots = &self.knots;
        let control_points = &self.control_points;
        let degree_inc = target_degree - self.degree;

        //intermediate values
        let mut bezalfs = vec![vec![T::zero(); new_degree + 1]; new_degree + degree_inc + 1];
        let new_control_point_count = control_points.len() + degree_inc + 3;
        let mut bpts = vec![OPoint::origin(); new_control_point_count];
        let mut e_bpts = vec![OPoint::origin(); new_control_point_count];
        let mut next_bpts = vec![OPoint::origin(); new_control_point_count];

        let m = n + new_degree + 1;
        let ph = target_degree;
        let ph2: usize = (T::from_usize(ph).unwrap() / T::from_f64(2.0).unwrap())
            .floor()
            .to_usize()
            .unwrap();

        let mut q_w = vec![OPoint::origin(); new_control_point_count];
        let mut u_h = vec![T::zero(); q_w.len() + target_degree + 1];

        bezalfs[0][0] = T::one();
        bezalfs[ph][new_degree] = T::one();

        let mut binom = Binomial::new();

        for i in 1..=ph2 {
            let inv = T::one() / binom.get(ph, i);
            let mpi = new_degree.min(i);
            for j in 0.max(i - degree_inc)..=mpi {
                bezalfs[i][j] = inv * binom.get(new_degree, j) * binom.get(degree_inc, i - j);
            }
        }

        for i in (ph2 + 1)..ph {
            let mpi = new_degree.min(i);
            for j in 0.max(i - degree_inc)..=mpi {
                bezalfs[i][j] = bezalfs[ph - i][new_degree - j];
            }
        }

        let mh = ph;
        let mut kind = ph + 1;
        let mut r: isize = -1;
        let mut a = new_degree;
        let mut b = new_degree + 1;
        let mut cind = 1;
        let mut ua = knots[0];
        q_w[0] = control_points[0].clone();
        for i in 0..=ph {
            u_h[i] = ua;
        }

        bpts[..(new_degree + 1)].clone_from_slice(&control_points[..(new_degree + 1)]);

        while b < m {
            let i = b;
            while b < m && knots[b] == knots[b + 1] {
                b += 1;
            }
            let mul = b - i + 1;
            let _mh = mh + mul + degree_inc;
            let ub = knots[b];
            let oldr = r;
            r = new_degree as isize - mul as isize;
            let lbz = if oldr > 0 {
                (T::from_isize(oldr + 2).unwrap() / T::from_f64(2.).unwrap())
                    .floor()
                    .to_usize()
                    .unwrap()
            } else {
                1
            };
            let rbz = if r > 0 {
                (T::from_usize(ph).unwrap()
                    - T::from_isize(r + 1).unwrap() / T::from_f64(2.).unwrap())
                .floor()
                .to_usize()
                .unwrap()
            } else {
                ph
            };
            if r > 0 {
                let numer = ub - ua;
                let mut alfs = vec![T::zero(); new_degree];
                let mut k = new_degree;
                while k > mul {
                    alfs[k - mul - 1] = numer / (knots[a + k] - ua);
                    k -= 1;
                }
                for j in 1..=(r as usize) {
                    let save = (r as usize) - j;
                    let s = mul + j;
                    let mut k = new_degree;
                    while k >= s {
                        bpts[k] = bpts[k].lerp(&bpts[k - 1], T::one() - alfs[k - s]);
                        k -= 1;
                    }
                    next_bpts[save] = bpts[new_degree].clone();
                }
            }

            for i in lbz..=ph {
                e_bpts[i] = OPoint::origin();
                let mpi = new_degree.min(i);
                for j in 0.max(i - degree_inc)..=mpi {
                    e_bpts[i].coords = &e_bpts[i].coords + &bpts[j].coords * bezalfs[i][j];
                }
            }

            if oldr > 1 {
                let mut first = kind - 2;
                let mut last = kind;
                let den = ub - ua;
                let bet = (ub - u_h[kind - 1]) / den;
                for tr in 1..oldr {
                    let mut i = first;
                    let mut j = last;
                    let mut kj = j - kind + 1;
                    let utr = tr as usize;
                    while (j as isize - i as isize) > tr {
                        if i < cind {
                            let alf = (ub - u_h[i]) / (ua - u_h[i]);
                            q_w[i] = q_w[i].lerp(&q_w[i - 1], T::one() - alf);
                        }
                        if j >= lbz {
                            if (j as isize) - tr <= (kind as isize - ph as isize + oldr) {
                                let gam = (ub - u_h[j - utr]) / den;
                                e_bpts[kj] = e_bpts[kj].lerp(&e_bpts[kj + 1], T::one() - gam);
                            }
                        } else {
                            e_bpts[kj] = e_bpts[kj].lerp(&e_bpts[kj + 1], T::one() - bet);
                        }
                        i += 1;
                        j -= 1;
                        kj -= 1;
                    }
                    first -= 1;
                    last += 1;
                }
            }

            if a != new_degree {
                for _i in 0..(ph as isize - oldr) {
                    u_h[kind] = ua;
                    kind += 1;
                }
            }

            for j in lbz..=rbz {
                q_w[cind] = e_bpts[j].clone();
                cind += 1;
            }

            if b < m {
                let ur = r as usize;
                bpts[..ur].clone_from_slice(&next_bpts[..ur]);
                for j in (r as usize)..=new_degree {
                    bpts[j] = control_points[b - new_degree + j].clone();
                }
                a = b;
                b += 1;
                ua = ub;
            } else {
                for i in 0..=ph {
                    u_h[kind + i] = ub;
                }
            }
        }

        Ok(Self {
            degree: target_degree,
            control_points: q_w,
            knots: KnotVector::new(u_h),
        })
    }

    /// Try to refine the curve by inserting knots
    pub fn knot_refine(&mut self, knots_to_insert: Vec<T>) {
        if knots_to_insert.is_empty() {
            return;
        }

        let degree = self.degree;
        let control_points = &self.control_points;

        let n = control_points.len() - 1;
        let m = n + degree + 1;
        let r = knots_to_insert.len() - 1;
        let a = self
            .knots
            .find_knot_span_index(n, degree, knots_to_insert[0]);
        let b = self
            .knots
            .find_knot_span_index(n, degree, knots_to_insert[r]);

        let mut control_points_post = vec![OPoint::<T, D>::origin(); n + 1 + r + 1];
        let mut knots_post = vec![T::zero(); m + 1 + r + 1];

        // dbg!(knots_to_insert.len(), &control_points_post.len(), a - degree + 1, b - 1, n);

        control_points_post[..((a - degree) + 1)]
            .clone_from_slice(&control_points[..((a - degree) + 1)]);
        for i in (b - 1)..=n {
            control_points_post[i + r + 1] = control_points[i].clone();
        }

        for i in 0..=a {
            knots_post[i] = self.knots[i];
        }
        for i in (b + degree)..=m {
            knots_post[i + r + 1] = self.knots[i];
        }

        let mut i = b + degree - 1;
        let mut k = b + degree + r;
        let mut j = r as isize;
        while j >= 0 {
            let uj = j as usize;
            while knots_to_insert[uj] <= self.knots[i] && i > a {
                control_points_post[k - degree - 1] = control_points[i - degree - 1].clone();
                knots_post[k] = self.knots[i];
                k -= 1;
                i -= 1;
            }
            control_points_post[k - degree - 1] = control_points_post[k - degree].clone();
            for l in 1..=degree {
                let ind = k - degree + l;
                let alpha = knots_post[k + l] - knots_to_insert[uj];
                if alpha.abs() < T::default_epsilon() {
                    control_points_post[ind - 1] = control_points_post[ind].clone();
                } else {
                    let weight = alpha / (knots_post[k + l] - self.knots[i - degree + l]);
                    control_points_post[ind - 1] = control_points_post[ind - 1]
                        .lerp(&control_points_post[ind], T::one() - weight);
                }
            }
            knots_post[k] = knots_to_insert[uj];
            k -= 1;
            j -= 1;
        }

        self.knots = KnotVector::new(knots_post);
        self.control_points = control_points_post;
    }
}

/// Enable to transform a NURBS curve by 3x3 matrix
impl<T: FloatingPoint> Transformable<&Matrix3<T>> for NurbsCurve2D<T> {
    fn transform(&mut self, transform: &Matrix3<T>) {
        self.control_points.iter_mut().for_each(|p| {
            let hom = Point2::new(p.x, p.y).to_homogeneous();
            let transformed = transform * hom;
            p.x = transformed.x;
            p.y = transformed.y;
        });
    }
}

/// Enable to transform a NURBS curve by 4x4 matrix
impl<T: FloatingPoint> Transformable<&Matrix4<T>> for NurbsCurve3D<T> {
    fn transform(&mut self, transform: &Matrix4<T>) {
        self.control_points.iter_mut().for_each(|p| {
            let hom = Point3::new(p.x, p.y, p.z).to_homogeneous();
            let transformed = transform * hom;
            p.x = transformed.x;
            p.y = transformed.y;
            p.z = transformed.z;
        });
    }
}

impl<T: FloatingPoint, D: DimName> Invertible for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<T, D>,
{
    /// Reverse the direction of the curve
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point2;
    /// use approx::assert_relative_eq;
    /// let points = vec![
    ///     Point2::new(0.0, 0.0),
    ///     Point2::new(1.0, 0.0),
    ///     Point2::new(1.0, 1.0),
    ///     Point2::new(0.0, 1.0),
    /// ];
    /// let mut curve = NurbsCurve2D::try_interpolate(&points, 3, None, None).unwrap();
    /// curve.invert();
    /// let (start, end) = curve.knots_domain();
    /// assert_relative_eq!(curve.point_at(start), points[points.len() - 1]);
    /// assert_relative_eq!(curve.point_at(end), points[0]);
    /// ```
    fn invert(&mut self) {
        self.control_points.reverse();
        self.knots.invert();
    }
}

/// Dehomogenize a point
pub fn dehomogenize<T: FloatingPoint, D: DimName>(
    point: &OPoint<T, D>,
) -> Option<OPoint<T, DimNameDiff<D, U1>>>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<T, D>,
    DefaultAllocator: Allocator<T, DimNameDiff<D, U1>>,
{
    let v = &point.coords;
    let idx = D::dim() - 1;
    let w = v[idx];
    if w != T::zero() {
        let coords =
            v.generic_view((0, 0), (<D as DimNameSub<U1>>::Output::name(), Const::<1>)) / w;
        Some(OPoint { coords })
    } else {
        None
    }
}
