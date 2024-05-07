use std::ops::Index;

use nalgebra::{convert, RealField};
use simba::scalar::SupersetOf;

use crate::{
    prelude::{Invertible, KnotMultiplicity},
    FloatingPoint,
};

/// Knot vector representation
#[derive(Clone, Debug)]
pub struct KnotVector<T> {
    knots: Vec<T>,
}

impl<T: RealField + Copy> KnotVector<T> {
    pub fn new(knots: Vec<T>) -> Self {
        Self { knots }
    }

    pub fn len(&self) -> usize {
        self.knots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.knots.is_empty()
    }

    pub fn first(&self) -> T {
        self.knots[0]
    }

    pub fn last(&self) -> T {
        self.knots[self.knots.len() - 1]
    }

    pub fn as_slice(&self) -> &[T] {
        &self.knots
    }

    pub fn iter(&self) -> std::slice::Iter<T> {
        self.knots.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.knots.iter_mut()
    }

    /// Get the domain of the knot vector by degree
    pub fn domain(&self, degree: usize) -> (T, T) {
        (
            self.knots[degree],
            self.knots[self.knots.len() - 1 - degree],
        )
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

        let mut current = KnotMultiplicity::new(self.knots[0], 0);
        self.knots.iter().for_each(|knot| {
            if (*knot - *current.knot()).abs() > T::default_epsilon() {
                mult.push(current.clone());
                current = KnotMultiplicity::new(*knot, 0);
            }
            current.increment_multiplicity();
        });
        mult.push(current);

        mult
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
        if u > self.knots[n + 1] - T::default_epsilon() {
            return n;
        }

        if u < self.knots[degree] + T::default_epsilon() {
            return degree;
        }

        // binary search
        let mut low = degree;
        let mut high = n + 1;
        let mut mid = ((low + high) as f64 / 2.).floor() as usize;
        while u < self.knots[mid] || self.knots[mid + 1] <= u {
            if u < self.knots[mid] {
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
            left[j] = u - self.knots[knot_span_index + 1 - j];
            right[j] = self.knots[knot_span_index + j] - u;
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
            left[j] = u - self.knots[knot_index + 1 - j];
            right[j] = self.knots[knot_index + j] - u;

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
            while u >= self.knots[knot_index + 1] && knot_index < n {
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
            while u >= self.knots[knot_index + 1] && knot_index < n {
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
        let n = self.knots.len() - degree - 2;
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
        KnotVector::new(self.knots.iter().map(|v| convert(*v)).collect())
    }
}

impl<T> Index<usize> for KnotVector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.knots[index]
    }
}

impl<T> FromIterator<T> for KnotVector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            knots: iter.into_iter().collect(),
        }
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
        let min = self.knots.first().unwrap();

        let mut next = vec![*min];
        let len = self.knots.len();
        for i in 1..len {
            next.push(next[i - 1] + (self.knots[len - i] - self.knots[len - i - 1]));
        }

        self.knots = next;
    }
}

#[cfg(test)]
mod tests {
    use super::KnotVector;

    #[test]
    fn knot() {
        let knot = KnotVector::new(vec![0., 0., 0., 1., 2., 3., 3., 3.]);
        let index = knot.find_knot_span_index(6, 2, 2.5);
        dbg!(index);
    }
}
