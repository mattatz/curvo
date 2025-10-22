use crate::misc::FloatingPoint;

/// Helper function to remove floating point fuzz from arc control point coordinates
/// Values very close to 0, 1, or -1 are snapped to exact values
pub fn arc_de_fuzz<T: FloatingPoint>(value: T) -> T {
    const FUZZ: f64 = 1.0e-6;
    let v = value.to_f64().unwrap();
    if v.abs() < FUZZ {
        T::zero()
    } else if (v - 1.0).abs() < FUZZ {
        T::one()
    } else if (v + 1.0).abs() < FUZZ {
        -T::one()
    } else {
        value
    }
}
