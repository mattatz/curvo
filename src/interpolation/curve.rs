use crate::{
    curve::{KnotStyle, NurbsCurve},
    knot::KnotVector,
    misc::FloatingPoint,
    prelude::PeriodicInterpolation,
};
use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, DMatrix, DVector, DefaultAllocator, DimName, DimNameAdd, DimNameDiff,
    DimNameSub, OPoint, U1,
};

use super::Interpolation;

impl<T: FloatingPoint, D: DimName> Interpolation for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Input = Vec<OPoint<T, DimNameDiff<D, U1>>>;
    type Output = anyhow::Result<Self>;

    /// Interpolate a NURBS curve
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
    /// let interpolated = NurbsCurve3D::interpolate(&points, 3);
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
    fn interpolate(input: &Self::Input, degree: usize) -> Self::Output {
        let (control_points, knots) = try_interpolate_control_points(
            &input
                .iter()
                .map(|p| DVector::from_vec(p.iter().copied().collect()))
                .collect_vec(),
            degree,
            true,
        )?;
        Ok(Self::new_unchecked(
            degree,
            control_points
                .iter()
                .map(|v| OPoint::from_slice(v.as_slice()))
                .collect(),
            knots,
        ))
    }
}

impl<T: FloatingPoint, D: DimName> PeriodicInterpolation for NurbsCurve<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <D as DimNameSub<U1>>::Output: DimNameAdd<U1>,
    DefaultAllocator: Allocator<<<D as DimNameSub<U1>>::Output as DimNameAdd<U1>>::Output>,
{
    type Input = Vec<OPoint<T, DimNameDiff<D, U1>>>;
    type Output = anyhow::Result<Self>;

    /// Try to create an periodic interpolated NURBS curve from a set of points
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// use nalgebra::Point3;
    /// use approx::assert_relative_eq;
    ///
    /// // Create periodic interpolated NURBS curve by a set of points
    /// let points: Vec<Point3<f64>> = vec![
    ///     Point3::new(-1.0, -1.0, 0.),
    ///     Point3::new(1.0, -1.0, 0.),
    ///     Point3::new(1.0, 1.0, 0.),
    ///     Point3::new(-1.0, 1.0, 0.),
    /// ];
    /// let closed = NurbsCurve3D::interpolate_periodic(&points, 3, KnotStyle::Centripetal).unwrap();
    ///
    /// let (start, end) = closed.knots_domain();
    ///
    /// // Check equality of the first and last points
    /// let head = closed.point_at(start);
    /// assert_relative_eq!(points[0], head, epsilon = 1e-10);
    ///
    /// let tail = closed.point_at(end);
    /// assert_relative_eq!(points[0], tail, epsilon = 1e-10);
    /// ```
    fn interpolate_periodic(
        input: &Self::Input,
        degree: usize,
        knot_style: KnotStyle,
    ) -> anyhow::Result<Self> {
        let input = input
            .iter()
            .map(|p| DVector::from_vec(p.iter().copied().collect()))
            .collect_vec();
        let (pts, knots) = interpolate_periodic_control_points(&input, degree, knot_style, true)?;

        Ok(Self::new_unchecked(
            degree,
            pts.iter()
                .map(|v| OPoint::from_slice(v.as_slice()))
                .collect(),
            knots,
        ))
    }
}

/// Try to interpolate the control points of a NURBS curve
pub fn try_interpolate_control_points<T: FloatingPoint>(
    points: &[DVector<T>],
    degree: usize,
    homogeneous: bool,
) -> anyhow::Result<(Vec<DVector<T>>, KnotVector<T>)> {
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

    let start = 1;
    let end = us.len() - degree;

    for i in start..end {
        let mut weight_sums = T::zero();
        for j in 0..degree {
            weight_sums += us[i + j];
        }
        knots_start.push(weight_sums / T::from_usize(degree).unwrap());
    }

    let knots = KnotVector::new([knots_start, vec![T::one(); degree + 1]].concat());
    let plen = points.len();

    let n = plen - 1;
    let ld = plen - (degree + 1);

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
        // println!("e = {:?}", &e);
        for j in 0..e.len() {
            m_a[(i, j)] = e[j];
        }
    }

    let control_points = try_solve_interpolation(m_a, points, homogeneous)?;

    Ok((control_points, knots))
}

/// Try to interpolate the control points of a periodic NURBS curve
pub fn interpolate_periodic_control_points<T: FloatingPoint>(
    points: &[DVector<T>],
    degree: usize,
    knot_style: KnotStyle,
    homogeneous: bool,
) -> anyhow::Result<(Vec<DVector<T>>, KnotVector<T>)> {
    let n = points.len();
    if n < degree + 1 {
        anyhow::bail!("Too few control points for curve");
    }

    // build periodic knot vector
    let parameters = knot_style.parameterize(points, true);

    let head = &parameters[0..(degree - 1)];
    let tail = &parameters[(parameters.len() - degree + 1)..];
    let start_parameters = tail.to_vec();

    let knots = [start_parameters, parameters.clone(), head.to_vec()].concat();

    let m = points.len() + degree;

    let knots = [
        vec![T::zero()],
        knots
            .iter()
            .scan(T::zero(), |p, x| {
                *p += *x;
                Some(*p)
            })
            .collect_vec(),
    ]
    .concat();

    // translate knot vectors to fit NURBS requirements
    let k0 = if degree > 2 {
        knots[0] - (knots[m + 1] - knots[m])
    } else {
        knots[0]
    };
    let k1 = if degree > 2 {
        knots[knots.len() - 1] + (knots[degree + 1] - knots[degree])
    } else {
        knots[knots.len() - 1]
    };
    let knots = [vec![k0], knots, vec![k1]].concat();

    let knots_vec = KnotVector::new(knots.clone());
    let plen = points.len();

    // build basis function coefficients matrix
    let mut m_a = DMatrix::<T>::zeros(plen, plen);
    let zero_pad = vec![T::zero(); plen - (degree + 1)];

    let n = knots_vec.len() - degree - 2;
    for i in 0..plen {
        let u = knots[i + degree]; // from start domain
        let knot_span_index = knots_vec.find_knot_span_index(n, degree, u);

        let basis = knots_vec.basis_functions(knot_span_index, u, degree);
        let basis_padded = [basis, zero_pad.clone()].concat();

        let ls = knot_span_index - degree;

        // In a closed periodic NURBS curve, the control points loop.
        // The solver only solves for the control points that are not looping,
        // so the coefficient matrix is looped to represent the duplicated control points.
        let n = basis_padded.len() - ls;
        let mut e = basis_padded[n..].to_vec();
        e.extend_from_slice(&basis_padded[0..n]);

        for j in 0..e.len() {
            m_a[(i, j)] = e[j];
        }
    }

    let mut control_points = try_solve_interpolation(m_a, points, homogeneous)?;

    // periodic
    for i in 0..(degree) {
        control_points.push(control_points[i].clone());
    }

    Ok((control_points, knots_vec))
}

/// Try to solve the interpolation problem
fn try_solve_interpolation<T: FloatingPoint>(
    m_a: DMatrix<T>,
    points: &[DVector<T>],
    homogeneous: bool,
) -> anyhow::Result<Vec<DVector<T>>> {
    let n = points.len();
    let dim = points[0].len();

    let lu = m_a.lu();
    // let lu = m_a.full_piv_lu();
    let mut m_x = DMatrix::<T>::identity(n, dim);
    let rows = m_x.nrows();
    for i in 0..dim {
        let b: Vec<_> = points.iter().map(|p| p[i]).collect();
        let b = DVector::from_vec(b);
        // println!("b = {:?}", &b);
        let xs = lu.solve(&b).ok_or(anyhow::anyhow!("Solve failed"))?;
        for j in 0..rows {
            m_x[(j, i)] = xs[j];
        }
    }

    // extract control points from solved x
    let mut control_points = vec![];
    for i in 0..m_x.nrows() {
        let mut coords = vec![];
        for j in 0..m_x.ncols() {
            coords.push(m_x[(i, j)]);
        }
        if homogeneous {
            coords.push(T::one());
        }
        control_points.push(DVector::from_vec(coords));
    }

    Ok(control_points)
}
