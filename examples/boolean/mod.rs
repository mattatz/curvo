pub mod scenario;
use curvo::prelude::{
    operation::BooleanOperation, Boolean, Clipped, CompoundCurve, CurveIntersectionSolverOptions,
    EndPoints, NurbsCurve, Region, Transformable,
};
use nalgebra::{Const, OMatrix, OPoint, U2, U3};
pub use scenario::*;

#[derive(Clone, Debug)]
pub enum CurveVariant {
    Curve(NurbsCurve<f64, Const<3>>),
    Compound(CompoundCurve<f64, Const<3>>),
    Region(Region<f64>),
}

impl<'a> Boolean<&'a CurveVariant> for CurveVariant {
    // type Output = anyhow::Result<Vec<Region<f64>>>;
    type Output = anyhow::Result<Clipped<f64>>;
    type Option = Option<CurveIntersectionSolverOptions<f64>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a CurveVariant,
        option: Self::Option,
    ) -> Self::Output {
        match (self, other) {
            (CurveVariant::Curve(s), CurveVariant::Curve(c)) => s.boolean(operation, c, option),
            (CurveVariant::Curve(_), CurveVariant::Compound(_)) => todo!(),
            (CurveVariant::Curve(_), CurveVariant::Region(_)) => todo!(),
            (CurveVariant::Compound(s), CurveVariant::Curve(c)) => s.boolean(operation, c, option),
            (CurveVariant::Compound(_), CurveVariant::Compound(_)) => todo!(),
            (CurveVariant::Compound(_), CurveVariant::Region(_)) => todo!(),
            (CurveVariant::Region(_), CurveVariant::Curve(_)) => todo!(),
            (CurveVariant::Region(_), CurveVariant::Compound(_)) => todo!(),
            (CurveVariant::Region(_), CurveVariant::Region(_)) => todo!(),
        }
    }
}

impl EndPoints<f64, U2> for CurveVariant {
    fn first_point(&self) -> OPoint<f64, U2> {
        match self {
            CurveVariant::Curve(c) => c.first_point(),
            CurveVariant::Compound(c) => c.first_point(),
            CurveVariant::Region(_) => todo!(),
        }
    }

    fn end_point(&self) -> OPoint<f64, U2> {
        match self {
            CurveVariant::Curve(c) => c.end_point(),
            CurveVariant::Compound(c) => c.end_point(),
            CurveVariant::Region(_) => todo!(),
        }
    }
}

impl From<NurbsCurve<f64, Const<3>>> for CurveVariant {
    fn from(value: NurbsCurve<f64, Const<3>>) -> Self {
        Self::Curve(value)
    }
}

impl From<CompoundCurve<f64, Const<3>>> for CurveVariant {
    fn from(value: CompoundCurve<f64, Const<3>>) -> Self {
        Self::Compound(value)
    }
}

impl From<Region<f64>> for CurveVariant {
    fn from(value: Region<f64>) -> Self {
        Self::Region(value)
    }
}

impl<'a> Transformable<&'a OMatrix<f64, U3, U3>> for CurveVariant {
    fn transform(&mut self, transform: &'a OMatrix<f64, U3, U3>) {
        match self {
            CurveVariant::Curve(c) => c.transform(transform),
            CurveVariant::Compound(c) => c.transform(transform),
            CurveVariant::Region(r) => r.transform(transform),
        }
    }
}
