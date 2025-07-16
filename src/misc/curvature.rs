use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, OVector};

use crate::misc::FloatingPoint;

/// Curvature structure
#[derive(Debug, Clone, PartialEq)]
pub struct Curvature<T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    /// Unit tangent vector
    t: OVector<T, D>,
    /// Curvature vector
    k: OVector<T, D>,
}

impl<T: FloatingPoint, D: DimName> Curvature<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(t: OVector<T, D>, k: OVector<T, D>) -> Self {
        Self { t, k }
    }

    /// Compute curvature from first and second derivatives
    /// Returns an error if the first derivative is zero
    pub fn derivatives(deriv1: OVector<T, D>, deriv2: OVector<T, D>) -> Result<Self, Self> {
        // Evaluate unit tangent and curvature from first and second derivatives
        // T = D1 / |D1|
        // K = ( D2 - (D2 o T)*T )/( D1 o D1)
        let n1 = deriv1.norm();
        if n1.is_zero() {
            // Use L'hopital's rule to show that if the unit tangent
            // exists and the 1rst derivative is zero and the 2nd derivative is
            // nonzero, then the unit tangent is equal to +/-the unitized
            // 2nd derivative.  The sign is equal to the sign of D1(s) o D2(s)
            // as s approaches the evaluation parameter.
            //
            let n2 = deriv2.norm();
            if !n2.is_zero() {
                Err(Self::new(OVector::zeros(), OVector::zeros()))
            } else {
                let u = deriv2 / n2;
                Err(Self::new(u, OVector::zeros()))
            }
        } else {
            let tangent = deriv1.clone() / n1;
            let dot = deriv2.dot(&tangent);
            let d1 = T::one() / (deriv1.dot(&deriv1));
            let k = (deriv2 - tangent.clone() * dot) * d1;
            Ok(Self::new(tangent, k))
        }
    }

    /// Returns the unit vector
    pub fn tangent_vector(&self) -> OVector<T, D> {
        self.t.clone()
    }

    /// Returns the curvature vector
    pub fn curvature_vector(&self) -> OVector<T, D> {
        self.k.clone()
    }

    /// Returns the curvature magnitude
    pub fn kappa(&self) -> T {
        self.k.norm()
    }
}
