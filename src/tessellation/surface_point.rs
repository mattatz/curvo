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
    u_constraint: bool,
    v_constraint: bool,
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
            u_constraint: false,
            v_constraint: false,
        }
    }

    pub fn with_u_constraint(mut self, u_constraint: bool) -> Self {
        self.u_constraint = u_constraint;
        self
    }

    pub fn with_v_constraint(mut self, v_constraint: bool) -> Self {
        self.v_constraint = v_constraint;
        self
    }

    pub fn is_u_constrained(&self) -> bool {
        self.u_constraint
    }

    pub fn is_v_constrained(&self) -> bool {
        self.v_constraint
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
