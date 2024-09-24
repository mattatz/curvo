use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOperation {
    Union,
    Intersection,
    Difference,
}

impl Display for BooleanOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BooleanOperation::Union => write!(f, "Union"),
            BooleanOperation::Intersection => write!(f, "Intersection"),
            BooleanOperation::Difference => write!(f, "Difference"),
        }
    }
}
