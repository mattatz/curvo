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
    pub is_normal_degenerated: bool,
}

impl<T: RealField, D: DimName> SurfacePoint<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    pub fn normal(&self) -> &OVector<T, D> {
        &self.normal
    }
}
