use nalgebra::{Matrix3, Vector3};

use crate::{misc::FloatingPoint, prelude::SurfaceTessellation3D};

/// A tangent space
#[derive(Debug, Clone)]
pub struct TangentSpace<T: FloatingPoint> {
    normal: Vector3<T>,
    tangent: Vector3<T>,
    bitangent: Vector3<T>,
}

impl<T: FloatingPoint> TangentSpace<T> {
    /// Create a new tangent space
    pub fn new(normal: Vector3<T>, tangent: Vector3<T>, bitangent: Vector3<T>) -> Self {
        Self {
            normal,
            tangent,
            bitangent,
        }
    }

    /// Get the normal
    pub fn normal(&self) -> &Vector3<T> {
        &self.normal
    }

    /// Get the tangent
    pub fn tangent(&self) -> &Vector3<T> {
        &self.tangent
    }

    /// Get the bitangent
    pub fn bitangent(&self) -> &Vector3<T> {
        &self.bitangent
    }

    /// Get the TBN matrix
    pub fn matrix(&self) -> Matrix3<T> {
        Matrix3::new(
            self.tangent.x,
            self.bitangent.x,
            self.normal.x,
            self.tangent.y,
            self.bitangent.y,
            self.normal.y,
            self.tangent.z,
            self.bitangent.z,
            self.normal.z,
        )
    }
}

impl<T: FloatingPoint> SurfaceTessellation3D<T> {
    /// Compute the tangent space for each vertex
    ///
    /// Returns a vector of TangentSpace for each vertex.
    pub fn compute_tangent_space(&self) -> Vec<TangentSpace<T>> {
        // Initialize accumulators for tangents and bitangents
        let mut tangents = vec![Vector3::zeros(); self.points.len()];
        let mut bitangents = vec![Vector3::zeros(); self.points.len()];

        // Process each triangle face
        for face in &self.faces {
            let i0 = face[0];
            let i1 = face[1];
            let i2 = face[2];

            // Get vertex positions
            let v0 = &self.points[i0];
            let v1 = &self.points[i1];
            let v2 = &self.points[i2];

            // Get UV coordinates
            let uv0 = &self.uvs[i0];
            let uv1 = &self.uvs[i1];
            let uv2 = &self.uvs[i2];

            // Calculate edge vectors
            let edge1 = v1 - v0;
            let edge2 = v2 - v0;

            // Calculate UV deltas
            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            // Calculate tangent and bitangent using the formula:
            // [T B] = 1/det * [deltaV2 -deltaV1] * [edge1]
            //                 [-deltaU2  deltaU1]   [edge2]
            let det = delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y;

            if det.abs() > T::default_epsilon() {
                let inv_det = T::one() / det;

                // Tangent = (deltaV2 * edge1 - deltaV1 * edge2) * inv_det
                let tangent = (edge1.scale(delta_uv2.y) - edge2.scale(delta_uv1.y)).scale(inv_det);

                // Bitangent = (-deltaU2 * edge1 + deltaU1 * edge2) * inv_det
                let bitangent =
                    (edge2.scale(delta_uv1.x) - edge1.scale(delta_uv2.x)).scale(inv_det);

                // Accumulate for each vertex of the triangle
                tangents[i0] += tangent;
                tangents[i1] += tangent;
                tangents[i2] += tangent;

                bitangents[i0] += bitangent;
                bitangents[i1] += bitangent;
                bitangents[i2] += bitangent;
            }
        }

        // Normalize and orthogonalize the tangents
        tangents
            .into_iter()
            .zip(bitangents.into_iter())
            .zip(self.normals.iter())
            .map(|((t, b), n)| {
                // Gram-Schmidt orthogonalization
                // T' = normalize(T - (T·N)N)
                let n_dot_t = n.dot(&t);
                let tangent = (t - n.scale(n_dot_t)).normalize();

                // Calculate handedness and adjust bitangent
                // B' = normalize(cross(N, T')) * handedness
                let computed_bitangent = n.cross(&tangent);
                let handedness = if computed_bitangent.dot(&b) < T::zero() {
                    -T::one()
                } else {
                    T::one()
                };
                let bitangent = computed_bitangent.scale(handedness);

                TangentSpace {
                    normal: *n,
                    tangent,
                    bitangent,
                }
            })
            .collect()
    }
}
