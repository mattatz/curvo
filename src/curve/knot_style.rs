use itertools::Itertools;
use nalgebra::DVector;

use crate::misc::FloatingPoint;

/// Knot parameterization for points interpolation
/// https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
#[derive(Debug, Clone)]
pub enum KnotStyle {
    Uniform,
    Chordal,
    Centripetal,
}

impl KnotStyle {
    pub fn parameterize<T: FloatingPoint>(&self, points: &[DVector<T>], closed: bool) -> Vec<T> where
    {
        match self {
            KnotStyle::Uniform => {
                let n = if closed {
                    points.len()
                } else {
                    points.len() - 1
                };
                let inv = T::one() / T::from_usize(n - 1).unwrap();
                (0..n).map(|_| inv).collect()
            }
            KnotStyle::Chordal | KnotStyle::Centripetal => {
                let alpha = self.alpha();
                let params: Vec<T> = if closed {
                    points
                        .iter()
                        .circular_tuple_windows()
                        .map(|(a, b)| (b - a).norm().powf(alpha))
                        .collect()
                } else {
                    points
                        .iter()
                        .tuple_windows()
                        .map(|(a, b)| (b - a).norm().powf(alpha))
                        .collect()
                };

                // println!("params: {:?}", params);
                params

                // normalize params
                // let max = params.iter().fold(T::zero(), |acc, &x| acc.max(x));
                // params.iter().map(|&x| x / max).collect()
            }
        }
    }

    pub fn alpha<T: FloatingPoint>(&self) -> T {
        match self {
            KnotStyle::Chordal => T::one(),
            KnotStyle::Centripetal => T::from_f64(0.5).unwrap(),
            _ => unimplemented!(),
        }
    }
}
