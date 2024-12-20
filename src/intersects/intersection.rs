use super::{has_intersection::HasIntersection, HasIntersectionParameter};

/// A struct representing the intersection of two objects.
#[derive(Debug, Clone)]
pub struct Intersection<P, T0, T1> {
    /// The point & parameter of the first object at the intersection.
    a: (P, T0),
    /// The point & parameter of the second object at the intersection.
    b: (P, T1),
}

impl<P, T0, T1> Intersection<P, T0, T1> {
    pub fn new(a: (P, T0), b: (P, T1)) -> Self {
        Self { a, b }
    }

    pub fn as_tuple(self) -> ((P, T0), (P, T1)) {
        (self.a, self.b)
    }
}

impl<P, T0: Clone + Copy, T1: Clone + Copy> HasIntersectionParameter<T0, T1>
    for Intersection<P, T0, T1>
{
    fn a_parameter(&self) -> T0 {
        self.a.1
    }

    fn b_parameter(&self) -> T1 {
        self.b.1
    }
}

impl<P, T0: Clone + Copy, T1: Clone + Copy> HasIntersection<(P, T0), (P, T1), T0, T1>
    for Intersection<P, T0, T1>
{
    fn a(&self) -> &(P, T0) {
        &self.a
    }

    fn b(&self) -> &(P, T1) {
        &self.b
    }
}

/// A struct representing the intersection of surface & curve.
pub type SurfaceCurveIntersection<P, T> = Intersection<P, (T, T), T>;
