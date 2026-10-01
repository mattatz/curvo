use std::collections::HashMap;

use nalgebra::RealField;

/// Returns the binomial coefficient of `n` and `k`.
#[allow(unused)]
pub fn binomial(n: usize, k: usize) -> f64 {
    if k == 0 || k == n {
        return 1.;
    } else if n == 0 || k > n {
        return 0.;
    }

    let k = k.min(n - k);
    let mut r = 1.;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r
}

/// A memoized binomial coefficient calculator.
/// A memoization map is used to store previously calculated binomial coefficients.
pub struct Binomial<T> {
    memo: HashMap<usize, HashMap<usize, T>>,
}

impl<T: RealField + Copy> Default for Binomial<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: RealField + Copy> Binomial<T> {
    pub fn new() -> Self {
        Self {
            memo: HashMap::new(),
        }
    }

    /// Returns the binomial coefficient of `n` and `k` with memoization.
    pub fn get(&mut self, n: usize, k: usize) -> T {
        if k == 0 || k == n {
            return T::one();
        } else if n == 0 || k > n {
            return T::zero();
        }

        let k = k.min(n - k);

        if let Some(memoized) = self.memo(n, k) {
            return memoized;
        }

        let r = self.get(n - 1, k) + self.get(n - 1, k - 1);
        self.memoize(n, k, r);
        r
    }

    /// Returns the memoized binomial coefficient of `n` and `k`.
    fn memo(&self, n: usize, k: usize) -> Option<T> {
        if let Some(m) = self.memo.get(&n) {
            if let Some(&v) = m.get(&k) {
                return Some(v);
            }
        }
        None
    }

    /// Memoizes the binomial coefficient of `n` and `k`.
    fn memoize(&mut self, n: usize, k: usize, v: T) {
        if let Some(m) = self.memo.get_mut(&n) {
            m.insert(k, v);
        } else {
            let mut m = HashMap::new();
            m.insert(k, v);
            self.memo.insert(n, m);
        }
    }
}

/// The binomial coefficient `n` choose `k`, without the memo [`Binomial`] allocates.
///
/// Up to `n = 24` every coefficient is below 2²⁴, so it is computed exactly in integers and is
/// exactly what [`Binomial::get`]'s sums give in `f32` or `f64`. Beyond that it is
/// [`Binomial::get`] itself, so the two never disagree and nothing can overflow.
pub(crate) fn binomial_coefficient<T: RealField + Copy>(n: usize, k: usize) -> T {
    if k > n {
        return T::zero();
    }
    if n > 24 {
        return Binomial::new().get(n, k);
    }
    let k = k.min(n - k);
    let mut c: u64 = 1;
    for i in 0..k {
        c = c * (n - i) as u64 / (i + 1) as u64;
    }
    T::from_u64(c).unwrap()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_binomial() {
        assert_eq!(super::binomial(5, 0), 1.);
        assert_eq!(super::binomial(5, 1), 5.);
        assert_eq!(super::binomial(5, 2), 10.);
        assert_eq!(super::binomial(5, 3), 10.);
        assert_eq!(super::binomial(5, 4), 5.);
        assert_eq!(super::binomial(5, 5), 1.);
        assert_eq!(super::binomial(5, 6), 0.);
    }

    #[test]
    fn binomial_coefficient_agrees_with_the_memoized_one_bit_for_bit() {
        let mut memo64 = super::Binomial::<f64>::new();
        let mut memo32 = super::Binomial::<f32>::new();
        for n in 0..=60 {
            for k in 0..=n + 1 {
                let a: f64 = super::binomial_coefficient(n, k);
                let b: f32 = super::binomial_coefficient(n, k);
                assert_eq!(a.to_bits(), memo64.get(n, k).to_bits(), "{n} choose {k}");
                assert_eq!(b.to_bits(), memo32.get(n, k).to_bits(), "{n} choose {k}");
            }
        }
    }

    #[test]
    fn test_memoized_binomial() {
        let mut binomial = super::Binomial::<f64>::new();
        for n in 1..10 {
            for k in 1..=n {
                assert_eq!(binomial.get(n, k), crate::misc::binomial::binomial(n, k));
            }
        }
    }
}
