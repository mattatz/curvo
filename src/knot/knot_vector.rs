use std::ops::Index;

use nalgebra::{convert, RealField};
use simba::scalar::SupersetOf;

use crate::{
    knot::KnotSide,
    prelude::{FloatingPoint, Invertible, KnotMultiplicity},
};

/// Knot vector representation
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KnotVector<T>(Vec<T>);

impl<T: RealField + Copy> KnotVector<T> {
    pub fn new(knots: Vec<T>) -> Self {
        Self(knots)
    }

    /// Create an uniform knot vector
    /// an uniform knot vector has a degree + 1 multiplicity at the start and end
    /// # Example
    /// ```
    /// use curvo::prelude::KnotVector;
    /// let knots: KnotVector<f64> = KnotVector::uniform(3, 2);
    /// assert_eq!(knots.to_vec(), vec![0., 0., 0., 1., 2., 2., 2.]);
    /// ```
    pub fn uniform(n: usize, degree: usize) -> Self {
        let mut knots = vec![];
        let m = degree;
        knots.extend(std::iter::repeat_n(T::zero(), m));
        for i in 0..n {
            knots.push(T::from_usize(i).unwrap());
        }
        knots.extend(std::iter::repeat_n(T::from_usize(n - 1).unwrap(), m));
        Self(knots)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.0.clone()
    }

    pub fn first(&self) -> T {
        self.0[0]
    }

    pub fn last(&self) -> T {
        self.0[self.0.len() - 1]
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn iter(&'_ self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }

    pub fn iter_mut(&'_ mut self) -> std::slice::IterMut<'_, T> {
        self.0.iter_mut()
    }

    /// Get the domain of the knot vector by degree
    pub fn domain(&self, degree: usize) -> (T, T) {
        (self.0[degree], self.0[self.0.len() - 1 - degree])
    }

    pub fn clamp(&self, degree: usize, u: T) -> T {
        let (min, max) = self.domain(degree);
        u.clamp(min, max)
    }

    /// Returns the index of the first knot greater than or equal to knot
    pub fn floor(&self, knot: T) -> Option<usize> {
        self.iter().rposition(|t| *t <= knot)
    }

    /// Add a knot and return the index of added knot
    pub fn add(&mut self, knot: T) -> usize {
        match self.floor(knot) {
            Some(idx) => {
                self.0.insert(idx + 1, knot);
                idx + 1
            }
            None => {
                self.0.insert(0, knot);
                0
            }
        }
    }

    /// Get the multiplicity of each knot
    /// # Example
    /// ```
    /// use curvo::prelude::KnotVector;
    /// let knots = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
    /// let knot_multiplicity = knots.multiplicity();
    /// assert_eq!(knot_multiplicity[0].multiplicity(), 3);
    /// assert_eq!(knot_multiplicity[1].multiplicity(), 1);
    /// assert_eq!(knot_multiplicity[2].multiplicity(), 1);
    /// assert_eq!(knot_multiplicity[3].multiplicity(), 3);
    /// ```
    pub fn multiplicity(&self) -> Vec<KnotMultiplicity<T>> {
        let mut mult = vec![];

        let mut current = KnotMultiplicity::new(self.0[0], 0);
        self.0.iter().for_each(|knot| {
            if (*knot - *current.knot()).abs() > T::default_epsilon() {
                mult.push(current.clone());
                current = KnotMultiplicity::new(*knot, 0);
            }
            current.increment_multiplicity();
        });
        mult.push(current);

        mult
    }

    /// Check if the knot vector is clamped
    /// `clamped` means the first and last knots have a multiplicity greater than the degree
    /// e.g. [0, 0, 0, 1, 2, 3, 3, 3] with degree 2 is clamped
    pub fn is_clamped(&self, degree: usize) -> bool {
        let multiplicity = self.multiplicity();
        let start = multiplicity.first();
        let end = multiplicity.last();
        match (start, end) {
            (Some(start), Some(end)) => {
                start.multiplicity() > degree && end.multiplicity() > degree
            }
            _ => false,
        }
    }

    /// Find the knot span index by linear search
    pub fn find_knot_span_linear(&self, n: usize, degree: usize, u: T) -> usize {
        let mut span = degree + 1;
        while span < n && u >= self.0[span] {
            span += 1;
        }
        span - 1
    }

    /// Find the knot span index by binary search
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::KnotVector;
    /// let knots = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
    /// let idx = knots.find_knot_span_index(6, 2, 2.5);
    /// assert_eq!(idx, 4);
    /// ```
    pub fn find_knot_span_index(&self, n: usize, degree: usize, u: T) -> usize {
        if u > self[n + 1] - T::default_epsilon() {
            return n;
        }

        if u < self[degree] + T::default_epsilon() {
            return degree;
        }

        // binary search
        self.binary_search(degree, n + 1, u)
    }

    /// Find the knot span index with a hint
    /// # Example
    /// ```
    /// use curvo::prelude::{KnotVector, KnotSide};
    /// let knots = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3., 4., 5., 5., 5.]);
    /// let degree = 2;
    /// let n = knots.len() - degree - 1;
    /// let t = 3.0;
    /// let idx = knots.find_knot_span_index_with_hint(n, degree, t, KnotSide::Below, None);
    /// assert_eq!(idx, 4);
    ///
    /// let idx = knots.find_knot_span_index_with_hint(n, degree, t, KnotSide::Above, None);
    /// assert_eq!(idx, 8);
    ///
    /// let idx = knots.find_knot_span_index_with_hint(n, degree, t, KnotSide::None, None);
    /// assert_eq!(idx, 7);
    /// ```
    pub fn find_knot_span_index_with_hint(
        &self,
        n: usize,
        degree: usize,
        u: T,
        side: KnotSide,
        hint: Option<usize>,
    ) -> usize {
        let mut low = degree;
        let mut high = n + 1;

        if let Some(h0) = hint {
            if h0 > degree && h0 < n + 1 {
                let mut h = h0;
                // move to the first index of a possible multiple-knot block
                while h > degree && self[h - 1] == self[h] {
                    h -= 1;
                }
                if h > degree {
                    if u < self[h] {
                        high = h;
                    } else {
                        if matches!(side, KnotSide::Below) && u == self[h] {
                            // left limit when u hits an internal knot
                            h = h.saturating_sub(1);
                        }
                        if h < degree {
                            h = degree;
                        }
                        low = h;
                    }
                }
            }
        }

        let mut idx = self.binary_search(low, high, u);
        match side {
            KnotSide::Below => {
                while idx > 0 && u == self[idx] {
                    idx -= 1;
                }
            }
            KnotSide::Above => {
                while idx < n && u == self[idx] {
                    idx += 1;
                }
            }
            _ => {}
        }
        idx
    }

    /// Find the knot span index with a binary search
    fn binary_search(&self, start: usize, end: usize, u: T) -> usize {
        let mut low = start;
        let mut high = end;
        let mut mid = ((low + high) as f64 / 2.).floor() as usize;
        while u < self[mid] || self[mid + 1] <= u {
            if u < self[mid] {
                high = mid;
            } else {
                low = mid;
            }
            let next = ((low + high) as f64 / 2.).floor() as usize;
            if mid == next {
                break;
            }
            mid = next;
        }
        mid
    }

    /// Compute the non-vanishing basis functions
    ///
    pub fn basis_functions(&self, knot_span_index: usize, u: T, degree: usize) -> Vec<T> {
        let mut basis_functions = vec![T::zero(); degree + 1];
        let mut left = vec![T::zero(); degree + 1];
        let mut right = vec![T::zero(); degree + 1];

        basis_functions[0] = T::one();

        for j in 1..=degree {
            left[j] = u - self[knot_span_index + 1 - j];
            right[j] = self[knot_span_index + j] - u;
            let mut saved = T::zero();

            for r in 0..j {
                let temp = basis_functions[r] / (right[r + 1] + left[j - r]);
                basis_functions[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }

            basis_functions[j] = saved;
        }

        basis_functions
    }

    /// Compute the non-vanishing basis functions and their derivatives
    /// 2d array of basis and derivative values of size (n+1, p+1) The nth row is the nth derivative and the first row is made up of the basis function values.
    pub fn derivative_basis_functions(
        &self,
        knot_index: usize,
        u: T,
        degree: usize,
        n: usize, // integer number of basis functions - 1 = knots.length - degree - 2
    ) -> Vec<Vec<T>> {
        let mut ndu = vec![vec![T::zero(); degree + 1]; degree + 1];
        let mut left = vec![T::zero(); degree + 1];
        let mut right = vec![T::zero(); degree + 1];

        ndu[0][0] = T::one();

        for j in 1..=degree {
            left[j] = u - self[knot_index + 1 - j];
            right[j] = self[knot_index + j] - u;

            let mut saved = T::zero();
            for r in 0..j {
                // lower triangle
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = ndu[r][j - 1] / ndu[j][r];

                // upper triangle
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }

        let mut ders = vec![vec![T::zero(); degree + 1]; n + 1];
        let mut a = vec![vec![T::zero(); degree + 1]; 2];

        // load the basis functions
        for j in 0..=degree {
            ders[0][j] = ndu[j][degree];
        }

        let idegree = degree as isize;
        let n = n as isize;

        // compute the derivatives
        for r in 0..=idegree {
            // alternate rows in array a
            let mut s1 = 0;
            let mut s2 = 1;
            a[0][0] = T::one();

            // loop to compute the kth derivative
            for k in 1..=n {
                let mut d = T::zero();
                let rk = r - k;
                let pk = idegree - k;

                if r >= k {
                    a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                    d = a[s2][0] * ndu[rk as usize][pk as usize];
                }

                let j1 = if rk >= -1 { 1 } else { -rk };
                let j2 = if r - 1 <= pk { k - 1 } else { idegree - r };

                for j in j1..=j2 {
                    a[s2][j as usize] = (a[s1][j as usize] - a[s1][j as usize - 1])
                        / ndu[(pk + 1) as usize][(rk + j) as usize];
                    d += a[s2][j as usize] * ndu[(rk + j) as usize][pk as usize];
                }

                let uk = k as usize;
                let ur = r as usize;
                if r <= pk {
                    a[s2][uk] = -a[s1][(k - 1) as usize] / ndu[(pk + 1) as usize][ur];
                    d += a[s2][uk] * ndu[ur][pk as usize];
                }

                ders[uk][ur] = d;

                // switch rows
                std::mem::swap(&mut s1, &mut s2);
            }
        }

        let mut acc = idegree;
        for k in 1..=n {
            for j in 0..=idegree {
                ders[k as usize][j as usize] *= T::from_isize(acc).unwrap();
            }
            acc *= idegree - k;
        }
        ders
    }

    /// Compute a regularly spaced basis functions
    /// Returns a tuple of knot spans and basis functions
    pub fn regulary_spaced_basis_functions(
        &self,
        degree: usize,
        divs: usize,
    ) -> (Vec<usize>, Vec<Vec<T>>) {
        let (start, _end, span, n) = self.regularly_spaced_span(degree, divs);

        let mut bases = vec![];
        let mut knot_spans = vec![];
        let mut u = start;
        let mut knot_index = self.find_knot_span_index(n, degree, u);

        // compute all of the basis functions
        for _i in 0..=divs {
            while u >= self[knot_index + 1] && knot_index < n {
                knot_index += 1;
            }
            knot_spans.push(knot_index);
            bases.push(self.basis_functions(knot_index, u, degree));
            u += span;
        }

        (knot_spans, bases)
    }

    /// Compute a regularly spaced basis functions and their derivatives
    /// Returns a tuple of knot spans and basis functions
    pub fn regularly_spaced_derivative_basis_functions(
        &self,
        degree: usize,
        divs: usize,
    ) -> (Vec<usize>, Vec<Vec<Vec<T>>>) {
        let (start, _end, span, n) = self.regularly_spaced_span(degree, divs);

        let mut bases = vec![];
        let mut knot_spans = vec![];
        let mut u = start;
        let mut knot_index = self.find_knot_span_index(n, degree, u);

        // compute all of the basis functions
        for _i in 0..=divs {
            while u >= self[knot_index + 1] && knot_index < n {
                knot_index += 1;
            }
            knot_spans.push(knot_index);
            bases.push(self.derivative_basis_functions(knot_index, u, degree, n));
            u += span;
        }

        (knot_spans, bases)
    }

    /// Compute a regularly spaced span & domain with a given degree and number of divisions
    pub fn regularly_spaced_span(&self, degree: usize, divs: usize) -> (T, T, T, usize) {
        let n = self.len() - degree - 2;
        let (start, end) = self.domain(degree);
        let span = (end - start) / T::from_usize(divs).unwrap();
        (start, end, span, n)
    }

    /// Cast the knot vector to another floating point type
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// let knots: KnotVector<f64> = KnotVector::new(vec![1., 2., 3., 4., 5., 6.]);
    /// let knots2 = knots.cast::<f32>();
    /// assert_eq!(knots.first(), 1.0);
    /// ```
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> KnotVector<F> {
        KnotVector::new(self.0.iter().map(|v| convert(*v)).collect())
    }

    /// Calculate a single Greville abscissa for a given index
    ///
    /// # Arguments
    ///
    /// * `index` - The control point index
    /// * `order` - The order (degree + 1)
    ///
    /// # Returns
    ///
    /// The Greville parameter value for the control point at the given index
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// let knots = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
    /// let g = knots.single_greville_abscissa(0, 3); // index=0, order=3 (degree=2)
    /// assert_eq!(g, 0.0);
    /// ```
    pub fn single_greville_abscissa(&self, index: usize, order: usize) -> T {
        if order < 2 {
            return self[index];
        }

        // Extract order-1 (degree) knots starting from index
        let degree = order - 1;
        let knot_slice = &self.0[index..index + degree];

        // degree = 1 or fully multiple knot
        if degree == 1 || knot_slice[0] == knot_slice[degree - 1] {
            return knot_slice[0];
        }

        // Calculate g = (knot[i]+...+knot[i+degree-1])/degree
        let k0 = knot_slice[0];
        let k = knot_slice[degree / 2];
        let k1 = knot_slice[degree - 1];
        let tol = (k1 - k0) * T::default_epsilon().sqrt();
        let const_deg = T::from_usize(degree).unwrap();

        let g = (0..degree)
            .map(|i| knot_slice[i])
            .reduce(|a, b| a + b)
            .unwrap();
        let g = g / const_deg;

        // Set g to exact value when knot vector is uniform
        let two = T::from_f64(2.0).unwrap();
        if (two * k - (k0 + k1)).abs() <= tol
            && (g - k).abs() <= (g.abs() * T::default_epsilon().sqrt() + tol)
        {
            k
        } else {
            g
        }
    }

    /// Calculate all Greville abscissae for the knot vector
    ///
    /// # Arguments
    ///
    /// * `order` - The order (degree + 1)
    /// * `cv_count` - The number of control points
    ///
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// let knots = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
    /// let degree = 2;
    /// let cv_count = 5;
    /// let greville = knots.greville_abscissae(degree + 1, cv_count).unwrap();
    /// assert_eq!(greville.len(), cv_count);
    /// ```
    pub fn greville_abscissae(&self, order: usize, cv_count: usize) -> anyhow::Result<Vec<T>> {
        anyhow::ensure!(order >= 2, "Order must be at least 2");
        anyhow::ensure!(cv_count >= order, "Control point count must be >= order");
        anyhow::ensure!(
            self.0.len() >= cv_count + order,
            "Not enough knots: expected at least {}, got {}",
            cv_count + order,
            self.0.len()
        );

        let count = cv_count;
        let degree = order - 1;
        if order == 2 {
            // degree = 1 case
            Ok((1..=count).map(|i| self[i]).collect())
        } else {
            Ok((0..count)
                .map(|i| self.single_greville_abscissa(i + degree - 1, order))
                .collect())
        }
    }
}

impl<T> Index<usize> for KnotVector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> FromIterator<T> for KnotVector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<T> From<Vec<T>> for KnotVector<T> {
    fn from(value: Vec<T>) -> Self {
        Self(value)
    }
}

impl<T: FloatingPoint> Invertible for KnotVector<T> {
    /// Reverses the knot vector
    /// # Example
    /// ```
    /// use curvo::prelude::*;
    /// let mut knot = KnotVector::new(vec![0., 0., 0., 1., 2., 2.5, 3.5, 4.0, 4.0]);
    /// knot.invert();
    ///
    /// let dst = vec![0.0, 0.0, 0.5, 1.5, 2.0, 3.0, 4.0, 4.0, 4.0];
    /// knot.iter().enumerate().for_each(|(i, v)| {
    ///     assert_eq!(*v, dst[i]);
    /// });
    /// ```
    fn invert(&mut self) {
        let min = self.0.first().unwrap();

        let mut next = vec![*min];
        let len = self.len();
        for i in 1..len {
            next.push(next[i - 1] + (self[len - i] - self[len - i - 1]));
        }

        self.0 = next;
    }
}

#[cfg(test)]
mod tests {
    use super::KnotVector;

    #[test]
    fn knot() {
        let knot = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
        let index = knot.find_knot_span_index(6, 2, 2.5);
        assert_eq!(index, 4);
    }
}
