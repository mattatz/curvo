use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint, OVector, U1,
};

use crate::{curve::nurbs_curve::dehomogenize, misc::FloatingPoint};

use super::NurbsSurface;

/// Evaluates one surface at many parameters, remembering the knot spans of the last ones.
///
/// Created by [`NurbsSurface::evaluator`]. Every method returns exactly what the surface's method
/// of the same name returns; what it saves is the binary search for each knot span when the
/// parameter lies in the span of the previous call or the one after it.
#[derive(Debug, Clone)]
pub struct SurfaceEvaluator<'a, T: FloatingPoint, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    surface: &'a NurbsSurface<T, D>,
    u_span: Option<usize>,
    v_span: Option<usize>,
}

impl<'a, T: FloatingPoint, D: DimName> SurfaceEvaluator<'a, T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub(crate) fn new(surface: &'a NurbsSurface<T, D>) -> Self {
        Self {
            surface,
            u_span: None,
            v_span: None,
        }
    }

    /// The surface this evaluates.
    pub fn surface(&self) -> &'a NurbsSurface<T, D> {
        self.surface
    }

    fn spans(&mut self, u: T, v: T) -> (usize, usize) {
        let (n, m) = self.surface.last_spans();
        let s = self.surface;
        (
            s.u_knots()
                .find_knot_span_index_cached(n, s.u_degree(), u, &mut self.u_span),
            s.v_knots()
                .find_knot_span_index_cached(m, s.v_degree(), v, &mut self.v_span),
        )
    }

    /// [`NurbsSurface::point_at`].
    pub fn point_at(&mut self, u: T, v: T) -> OPoint<T, DimNameDiff<D, U1>> {
        let (su, sv) = self.spans(u, v);
        dehomogenize(&self.surface.point_in_span(su, sv, u, v)).unwrap()
    }

    /// [`NurbsSurface::rational_derivatives`].
    pub fn rational_derivatives(
        &mut self,
        u: T,
        v: T,
        derivs: usize,
    ) -> Vec<Vec<OVector<T, DimNameDiff<D, U1>>>> {
        let (su, sv) = self.spans(u, v);
        self.surface
            .rational_derivatives_in_span(su, sv, u, v, derivs)
    }

    /// [`NurbsSurface::normal_at`]: `dS/du × dS/dv`, not unit length.
    pub fn normal_at(&mut self, u: T, v: T) -> OVector<T, DimNameDiff<D, U1>> {
        let deriv = self.rational_derivatives(u, v, 1);
        let v0 = &deriv[1][0];
        let v1 = &deriv[0][1];
        v0.cross(v1)
    }
}
