use crate::misc::FloatingPoint;

mod curve;

/// The type of discontinuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiscontinuityType {
    /// C0 discontinuity.
    C0,
    /// C1 (1st derivative) discontinuity.
    C1,
    /// C2 (2nd derivative) discontinuity.
    C2,
}

/// Search for a derivative, tangent, or curvature discontinuity.
pub trait Discontinuity<T> {
    /// Get the next discontinuity.
    fn get_next_discontinuity(&self, ty: DiscontinuityType, start: T, end: T) -> Option<T>;

    /// Get an iterator for searching for discontinuities.
    fn discontinuity_iter(
        &self,
        ty: DiscontinuityType,
        start: T,
        end: T,
    ) -> DiscontinuityIterator<'_, Self, T>
    where
        Self: Sized,
    {
        DiscontinuityIterator {
            geometry: self,
            ty,
            next: start,
            end,
        }
    }
}

/// Iterator for searching for discontinuities.
pub struct DiscontinuityIterator<'a, G, T>
where
    G: Discontinuity<T>,
{
    geometry: &'a G,
    ty: DiscontinuityType,
    next: T,
    end: T,
}

impl<'a, G, T> Iterator for DiscontinuityIterator<'a, G, T>
where
    G: Discontinuity<T>,
    T: FloatingPoint,
{
    type Item = T;

    /// Get the next discontinuity.
    fn next(&mut self) -> Option<Self::Item> {
        let item = self
            .geometry
            .get_next_discontinuity(self.ty, self.next, self.end);
        if let Some(t) = item {
            let h = (self.end - self.next) * T::from_f64(1e-12).unwrap();
            self.next = t + h;
            // println!("next: {}, end: {}", self.next, self.end);
            Some(t)
        } else {
            None
        }
    }
}
