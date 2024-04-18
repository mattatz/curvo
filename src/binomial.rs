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
    fn test_memoized_binomial() {
        let mut binomial = super::Binomial::<f64>::new();
        for n in 1..10 {
            for k in 1..=n {
                assert_eq!(binomial.get(n, k), crate::binomial::binomial(n, k));
            }
        }
    }
}
