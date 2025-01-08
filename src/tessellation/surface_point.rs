use nalgebra::{
    allocator::Allocator, DefaultAllocator, DimName, DimNameSub, OPoint, OVector, RealField,
    Vector2, U1,
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
    // constraints to be divided
    u_constraint: bool,
    v_constraint: bool,
    // flags to indicate if this point is on the boundary or not
    u_min: bool,
    u_max: bool,
    v_min: bool,
    v_max: bool,
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
            u_min: false,
            u_max: false,
            v_min: false,
            v_max: false,
        }
    }

    pub fn with_constraints(mut self, u: bool, v: bool) -> Self {
        self.u_constraint = u;
        self.v_constraint = v;
        self
    }

    pub fn is_u_constrained(&self) -> bool {
        self.u_constraint
    }

    pub fn is_v_constrained(&self) -> bool {
        self.v_constraint
    }

    pub fn with_boundary(mut self, u_min: bool, u_max: bool, v_min: bool, v_max: bool) -> Self {
        self.u_min = u_min;
        self.u_max = u_max;
        self.v_min = v_min;
        self.v_max = v_max;
        self
    }

    pub fn is_u_min(&self) -> bool {
        self.u_min
    }

    pub fn is_u_max(&self) -> bool {
        self.u_max
    }

    pub fn is_v_min(&self) -> bool {
        self.v_min
    }

    pub fn is_v_max(&self) -> bool {
        self.v_max
    }

    pub fn into_tuple(self) -> (Vector2<T>, OPoint<T, D>, OVector<T, D>) {
        (self.uv, self.point, self.normal)
    }

    pub fn normal(&self) -> &OVector<T, D> {
        &self.normal
    }

    pub fn point(&self) -> &OPoint<T, D> {
        &self.point
    }

    pub fn uv(&self) -> &Vector2<T> {
        &self.uv
    }

    pub fn is_normal_degenerated(&self) -> bool {
        self.is_normal_degenerated
    }
}
