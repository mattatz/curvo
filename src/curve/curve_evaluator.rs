use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, U1,
};

use crate::{curve::nurbs_curve::dehomogenize, misc::FloatingPoint};

use super::NurbsCurve;

/// Evaluates one curve at many parameters, remembering the knot span of the last one.
///
/// Created by [`NurbsCurve::evaluator`]. Every method returns exactly what the curve's method of
/// the same name returns; what it saves is the binary search for the knot span when the parameter
/// lies in the span of the previous call or the one after it.
#[derive(Debug, Clone)]
pub struct CurveEvaluator<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    curve: &'a NurbsCurve<T, D>,
    span: Option<usize>,
}

impl<'a, T: FloatingPoint, D: DimName> CurveEvaluator<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub(crate) fn new(curve: &'a NurbsCurve<T, D>) -> Self {
        Self { curve, span: None }
    }

    /// The curve this evaluates.
    pub fn curve(&self) -> &'a NurbsCurve<T, D> {
        self.curve
    }

    fn span(&mut self, u: T) -> usize {
        self.curve.knots().find_knot_span_index_cached(
            self.curve.last_span(),
            self.curve.degree(),
            u,
            &mut self.span,
        )
    }

    /// [`NurbsCurve::point_at`].
    pub fn point_at(&mut self, t: T) -> OPoint<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let span = self.span(t);
        dehomogenize(&self.curve.point_in_span(span, t)).unwrap()
    }

    /// [`NurbsCurve::rational_derivatives`].
    pub fn rational_derivatives(
        &mut self,
        u: T,
        derivs: usize,
    ) -> Vec<OVector<T, DimNameDiff<D, U1>>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        let span = self.span(u);
        self.curve.rational_derivatives_in_span(span, u, derivs)
    }

    /// [`NurbsCurve::tangent_at`].
    pub fn tangent_at(&mut self, u: T) -> OVector<T, DimNameDiff<D, U1>>
    where
        D: DimNameSub<U1>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        self.rational_derivatives(u, 1)[1].clone()
    }
}
