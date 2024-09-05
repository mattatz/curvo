use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::{allocator::Allocator, Const, DefaultAllocator};

use crate::{
    boolean::{degeneracies::Degeneracy, node::Node, operation::BooleanOperation},
    curve::NurbsCurve,
    misc::FloatingPoint,
    prelude::{Contains, CurveIntersectionSolverOptions, HasIntersection},
    region::{CompoundCurve, Region},
};

use super::Boolean;

/// Boolean operation for compound curve & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, Const<3>>>
    for CompoundCurve<T, Const<3>>
where
    DefaultAllocator: Allocator<Const<3>>,
{
    type Output = anyhow::Result<Vec<Region<T>>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: super::operation::BooleanOperation,
        clip: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        let intersections = self.find_intersections(clip, option.clone())?;

        todo!()
    }
}
