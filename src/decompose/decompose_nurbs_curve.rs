use itertools::Itertools;
use nalgebra::{allocator::Allocator, DefaultAllocator, DimName, DimNameDiff, DimNameSub, U1};

use crate::{curve::NurbsCurve, knot::KnotVector, misc::FloatingPoint, prelude::Decompose};

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use approx::assert_relative_eq;
    use nalgebra::Point3;

    #[test]
    fn test_decompose_periodic_curve() {
        // Create a periodic curve with 8 control points and degree 3
        let points: Vec<Point3<f64>> = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(-1.0, 1.0, 0.0),
            Point3::new(-1.0, 0.0, 0.0),
            Point3::new(-1.0, -1.0, 0.0),
            Point3::new(0.0, -1.0, 0.0),
            Point3::new(1.0, -1.0, 0.0),
        ];

        let curve = NurbsCurve3D::try_periodic(&points, 3).unwrap();

        // Verify the curve properties
        assert_eq!(curve.degree(), 3);
        assert!(!curve.is_clamped());

        let (domain_start, domain_end) = curve.knots_domain();
        assert_eq!(domain_start, 3.0);
        assert_eq!(domain_end, 11.0);

        // Decompose the curve
        let segments = curve.try_decompose().unwrap();

        // Should have 8 segments (one for each knot span in the domain)
        assert_eq!(segments.len(), 8);

        // Verify each segment's domain is within the original curve's domain
        for (i, seg) in segments.iter().enumerate() {
            let (seg_start, seg_end) = seg.knots_domain();
            assert!(
                seg_start >= domain_start,
                "Segment {} start {} is before domain start {}",
                i,
                seg_start,
                domain_start
            );
            assert!(
                seg_end <= domain_end,
                "Segment {} end {} is after domain end {}",
                i,
                seg_end,
                domain_end
            );

            // Each segment should be a Bezier curve (clamped with degree+1 multiplicity at ends)
            assert!(seg.is_clamped());
            assert_eq!(seg.degree(), 3);
            assert_eq!(seg.control_points().len(), 4); // degree + 1 control points
        }

        // Verify segments cover the domain continuously
        let mut expected_start = domain_start;
        for seg in segments.iter() {
            let (seg_start, seg_end) = seg.knots_domain();
            assert_relative_eq!(seg_start, expected_start, epsilon = 1e-10);
            expected_start = seg_end;
        }
        assert_relative_eq!(expected_start, domain_end, epsilon = 1e-10);

        // Verify geometric continuity: end of segment i should match start of segment i+1
        for i in 0..segments.len() - 1 {
            let (_, end) = segments[i].knots_domain();
            let (start, _) = segments[i + 1].knots_domain();
            let p_end = segments[i].point_at(end);
            let p_start = segments[i + 1].point_at(start);
            assert_relative_eq!(p_end, p_start, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_decompose_clamped_curve() {
        // Create a clamped (interpolated) curve
        let points: Vec<Point3<f64>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(3.0, 1.0, 0.0),
            Point3::new(4.0, 3.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
        ];

        let curve: NurbsCurve3D<f64> = NurbsCurve3D::interpolate(&points, 3).unwrap();

        // Verify the curve properties
        assert_eq!(curve.degree(), 3);
        assert!(curve.is_clamped());

        let (domain_start, domain_end) = curve.knots_domain();

        // Decompose the curve
        let segments = curve.try_decompose().unwrap();

        // Should have 2 segments (5 points with degree 3 creates 2 internal knot spans)
        assert_eq!(segments.len(), 2);

        // Verify each segment is a valid Bezier curve
        for seg in segments.iter() {
            assert!(seg.is_clamped());
            assert_eq!(seg.degree(), 3);
            assert_eq!(seg.control_points().len(), 4);
        }

        // Verify segments cover the domain
        let (first_start, _) = segments.first().unwrap().knots_domain();
        let (_, last_end) = segments.last().unwrap().knots_domain();
        assert_relative_eq!(first_start, domain_start, epsilon = 1e-10);
        assert_relative_eq!(last_end, domain_end, epsilon = 1e-10);

        // Verify geometric continuity
        let (_, end) = segments[0].knots_domain();
        let (start, _) = segments[1].knots_domain();
        let p_end = segments[0].point_at(end);
        let p_start = segments[1].point_at(start);
        assert_relative_eq!(p_end, p_start, epsilon = 1e-10);
    }

    #[test]
    fn test_decompose_polyline() {
        // Create a simple polyline (degree 1)
        let points: Vec<Point3<f64>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];

        let curve = NurbsCurve3D::<f64>::polyline(&points, false);

        assert_eq!(curve.degree(), 1);
        assert!(curve.is_clamped());

        let segments = curve.try_decompose().unwrap();

        // Should have 3 segments (4 points creates 3 line segments)
        assert_eq!(segments.len(), 3);

        // Verify each segment
        for seg in segments.iter() {
            assert!(seg.is_clamped());
            assert_eq!(seg.degree(), 1);
            assert_eq!(seg.control_points().len(), 2); // degree + 1 = 2 control points
        }

        // Verify geometric continuity
        for i in 0..segments.len() - 1 {
            let (_, end) = segments[i].knots_domain();
            let (start, _) = segments[i + 1].knots_domain();
            let p_end = segments[i].point_at(end);
            let p_start = segments[i + 1].point_at(start);
            assert_relative_eq!(p_end, p_start, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_decompose_single_bezier() {
        // Create a single Bezier curve (already decomposed)
        let points: Vec<Point3<f64>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];

        let curve = NurbsCurve3D::<f64>::bezier(&points);

        assert_eq!(curve.degree(), 3);
        assert!(curve.is_clamped());

        let segments = curve.try_decompose().unwrap();

        // Should have 1 segment (already a single Bezier)
        assert_eq!(segments.len(), 1);

        let seg = &segments[0];
        assert!(seg.is_clamped());
        assert_eq!(seg.degree(), 3);
        assert_eq!(seg.control_points().len(), 4);
    }
}

impl<T: FloatingPoint, D: DimName> Decompose for NurbsCurve<T, D>
where
    DefaultAllocator: Allocator<D>,
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Output = Vec<NurbsCurve<T, D>>;

    /// Decompose the curve into a set of Bezier segments of the same degree
    fn try_decompose(&self) -> anyhow::Result<Self::Output> {
        let mut cloned = self.clone();
        let degree = cloned.degree();
        let req_mult = degree + 1;

        // Get the domain boundaries before any modification
        let (domain_start, domain_end) = cloned.knots_domain();

        // Get unique knots within the domain and count them
        let knot_mults = cloned.knots().multiplicity();
        let domain_knot_values: Vec<T> = knot_mults
            .iter()
            .filter(|m| *m.knot() >= domain_start && *m.knot() <= domain_end)
            .map(|m| *m.knot())
            .collect();
        let num_segments = if domain_knot_values.len() > 1 {
            domain_knot_values.len() - 1
        } else {
            1
        };

        // Insert knots to make each unique knot within domain have multiplicity degree + 1
        for knot_mult in knot_mults.iter() {
            let knot = *knot_mult.knot();
            if knot >= domain_start && knot <= domain_end && knot_mult.multiplicity() < req_mult {
                let knots_insert = vec![knot; req_mult - knot_mult.multiplicity()];
                cloned.try_refine_knot(knots_insert)?;
            }
        }

        if num_segments <= 1 {
            Ok(vec![cloned])
        } else {
            // Find the starting index in the refined knot vector for domain_start
            let knots_slice = cloned.knots().as_slice();
            let start_idx = knots_slice
                .iter()
                .position(|&k| k == domain_start)
                .unwrap_or(0);

            let knot_length = req_mult * 2;
            let segments = (0..num_segments)
                .map(|i| {
                    let idx = start_idx + i * req_mult;
                    let knots = cloned.knots().as_slice()[idx..(idx + knot_length)].to_vec();
                    let control_points = cloned.control_points()[idx..(idx + req_mult)].to_vec();
                    NurbsCurve::new_unchecked(
                        cloned.degree(),
                        control_points,
                        KnotVector::new(knots),
                    )
                })
                .collect_vec();
            Ok(segments)
        }
    }
}
