use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, OPoint, OVector, RealField, Vector2,
};

/// Surface point representation
/// containing evaluated data on a surface in adaptive tessellation processor
#[derive(Clone, Debug)]
pub struct SurfacePoint<T: RealField, D: DimName>
where
    DefaultAllocator: Allocator<D>,
{
    pub uv: Vector2<T>,
    pub point: OPoint<T, D>,
    pub normal: OVector<T, D>,
    is_normal_degenerated: bool,
}

impl<T: RealField, D: DimName> SurfacePoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn new(
        uv: Vector2<T>,
        point: OPoint<T, D>,
        normal: OVector<T, D>,
        is_normal_degenerated: bool,
    ) -> Self {
        Self {
            uv,
            point,
            normal,
            is_normal_degenerated,
        }
    }

    pub fn into_tuple(self) -> (Vector2<T>, OPoint<T, D>, OVector<T, D>) {
        (self.uv, self.point, self.normal)
    }

    pub fn normal(&self) -> &OVector<T, D> {
        &self.normal
    }

    pub fn is_normal_degenerated(&self) -> bool {
        self.is_normal_degenerated
    }
}
