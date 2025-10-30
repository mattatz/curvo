use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimName, DimNameDiff, DimNameSub, OPoint,
    OVector, Vector2, U1,
};
use ordered_float::OrderedFloat;
use simba::scalar::SupersetOf;

use crate::{
    misc::FloatingPoint, prelude::NurbsSurface,
    tessellation::adaptive_tessellation_node::AdaptiveTessellationNode,
};

use super::boundary_constraints::{BoundaryConstraints, BoundaryEvaluation};

/// Surface tessellation representation
/// This struct is used to create a mesh data from surface
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceTessellation<T: FloatingPoint, D: DimName>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    pub(crate) points: Vec<OPoint<T, DimNameDiff<D, U1>>>,
    pub(crate) normals: Vec<OVector<T, DimNameDiff<D, U1>>>,
    pub(crate) faces: Vec<[usize; 3]>,
    pub(crate) uvs: Vec<Vector2<T>>,
}

/// 2D tessellation alias
pub type SurfaceTessellation2D<T> = SurfaceTessellation<T, Const<3>>;

/// 3D tessellation alias
pub type SurfaceTessellation3D<T> = SurfaceTessellation<T, Const<4>>;

type HashKey = (OrderedFloat<f64>, OrderedFloat<f64>);

impl<T: FloatingPoint, D: DimName> SurfaceTessellation<T, D>
where
    D: DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    /// Create a new surface tessellation from surface and adaptive tessellation nodes
    pub fn new(
        surface: &NurbsSurface<T, D>,
        nodes: &Vec<AdaptiveTessellationNode<T, D>>,
        constraints: Option<BoundaryConstraints<T>>,
    ) -> Self {
        let mut tess = Self {
            points: Default::default(),
            normals: Default::default(),
            faces: Default::default(),
            uvs: Default::default(),
        };

        let boundary_evaluation = constraints.map(|c| BoundaryEvaluation::new(surface, &c));
        let mut map: HashMap<HashKey, usize> = HashMap::new();

        // Triangulate all nodes
        nodes.iter().for_each(|node| {
            if node.is_leaf() {
                tess.triangulate(&mut map, surface, nodes, node, boundary_evaluation.as_ref());
            }
        });

        tess
    }

    /// Triangulate the surface with adaptive tessellation nodes recursively
    fn triangulate(
        &mut self,
        map: &mut HashMap<HashKey, usize>,
        surface: &NurbsSurface<T, D>,
        nodes: &Vec<AdaptiveTessellationNode<T, D>>,
        leaf_node: &AdaptiveTessellationNode<T, D>,
        boundary_evaluation: Option<&BoundaryEvaluation<T, D>>,
    ) {
        let corners = (0..4)
            .map(|i| leaf_node.get_all_corners(nodes, i))
            .collect_vec();

        let split_id = corners
            .iter()
            .position(|c| c.len() == 2)
            .map(|i| i + 1)
            .unwrap_or(0);
        let pts = corners.into_iter().flatten().collect_vec();

        let pts = if let Some(boundary_evaluation) = boundary_evaluation {
            pts.into_iter()
                .map(|pt| {
                    if pt.is_u_min() {
                        if let Some(closest) = boundary_evaluation.closest_point_at_u_min(&pt) {
                            return closest;
                        }
                    }
                    if pt.is_u_max() {
                        if let Some(closest) = boundary_evaluation.closest_point_at_u_max(&pt) {
                            return closest;
                        }
                    }
                    if pt.is_v_min() {
                        if let Some(closest) = boundary_evaluation.closest_point_at_v_min(&pt) {
                            return closest;
                        }
                    }
                    if pt.is_v_max() {
                        if let Some(closest) = boundary_evaluation.closest_point_at_v_max(&pt) {
                            return closest;
                        }
                    }
                    pt
                })
                .collect_vec()
        } else {
            pts
        };

        let n = pts.len();
        let mut ids = Vec::with_capacity(n);
        for corner in pts.into_iter() {
            let uv = corner.uv();
            let key = (
                OrderedFloat::from(T::to_f64(&uv.x).unwrap()),
                OrderedFloat::from(T::to_f64(&uv.y).unwrap()),
            );
            let id = map.entry(key).or_insert_with(|| {
                let (uv, point, normal) = corner.into_tuple();
                self.points.push(point);
                self.normals.push(normal);
                self.uvs.push(uv);
                self.points.len() - 1
            });
            ids.push(*id);
        }

        match n {
            0 => {}
            4 => {
                self.faces.push([ids[0], ids[1], ids[3]]);
                self.faces.push([ids[3], ids[1], ids[2]]);
            }
            5 => {
                let il = ids.len();
                let a = ids[split_id];
                let b = ids[(split_id + 1) % il];
                let c = ids[(split_id + 2) % il];
                let d = ids[(split_id + 3) % il];
                let e = ids[(split_id + 4) % il];
                self.faces.push([a, b, c]);
                self.faces.push([a, d, e]);
                self.faces.push([a, c, d]);
            }
            m => {
                let center = leaf_node.center(surface);
                self.points.push(center.point.clone());
                self.normals.push(center.normal.clone());
                self.uvs.push(center.uv);

                let center_index = self.points.len() - 1;
                let mut j = m - 1;
                let mut i = 0;
                while i < m {
                    self.faces.push([center_index, ids[j], ids[i]]);
                    j = i;
                    i += 1;
                }
            }
        };
    }

    /// Get the points
    pub fn points(&self) -> &Vec<OPoint<T, DimNameDiff<D, U1>>> {
        &self.points
    }

    /// Get the normals
    pub fn normals(&self) -> &Vec<OVector<T, DimNameDiff<D, U1>>> {
        &self.normals
    }

    /// Get the uvs
    pub fn uvs(&self) -> &Vec<Vector2<T>> {
        &self.uvs
    }

    /// Zip the points, normals, and uvs together
    #[allow(clippy::type_complexity)]
    pub fn zipped_iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &OPoint<T, DimNameDiff<D, U1>>,
            &OVector<T, DimNameDiff<D, U1>>,
            &Vector2<T>,
        ),
    > {
        self.points
            .iter()
            .zip(self.normals.iter())
            .zip(self.uvs.iter())
            .map(|((p, n), uv)| (p, n, uv))
    }

    /// Zip the points, normals, and uvs together mutably
    #[allow(clippy::type_complexity)]
    pub fn zipped_iter_mut(
        &mut self,
    ) -> impl Iterator<
        Item = (
            &mut OPoint<T, DimNameDiff<D, U1>>,
            &mut OVector<T, DimNameDiff<D, U1>>,
            &mut Vector2<T>,
        ),
    > {
        self.points
            .iter_mut()
            .zip(self.normals.iter_mut())
            .zip(self.uvs.iter_mut())
            .map(|((p, n), uv)| (p, n, uv))
    }

    /// Get the faces
    pub fn faces(&self) -> &Vec<[usize; 3]> {
        &self.faces
    }

    /// Cast the surface tessellation to another floating point type.
    pub fn cast<F: FloatingPoint + SupersetOf<T>>(&self) -> SurfaceTessellation<F, D>
    where
        DefaultAllocator: Allocator<D>,
        DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    {
        SurfaceTessellation {
            points: self.points.iter().map(|p| p.clone().cast()).collect(),
            normals: self.normals.iter().map(|n| n.clone().cast()).collect(),
            faces: self.faces.clone(),
            uvs: self.uvs.iter().map(|uv| uv.cast()).collect(),
        }
    }
}

#[cfg(feature = "serde")]
impl<T, D> serde::Serialize for SurfaceTessellation<T, D>
where
    T: FloatingPoint + serde::Serialize,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <DefaultAllocator as nalgebra::allocator::Allocator<DimNameDiff<D, U1>>>::Buffer<T>:
        serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("SurfaceTessellation", 4)?;
        state.serialize_field("points", &self.points)?;
        state.serialize_field("normals", &self.normals)?;
        state.serialize_field("faces", &self.faces)?;
        state.serialize_field("uvs", &self.uvs)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de, T, D> serde::Deserialize<'de> for SurfaceTessellation<T, D>
where
    T: FloatingPoint + serde::Deserialize<'de>,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
    <DefaultAllocator as nalgebra::allocator::Allocator<DimNameDiff<D, U1>>>::Buffer<T>:
        serde::Deserialize<'de>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {
            Points,
            Normals,
            Faces,
            Uvs,
        }

        struct SurfaceTessellationVisitor<T, D>
        where
            T: FloatingPoint,
            D: DimName + DimNameSub<U1>,
            DefaultAllocator: Allocator<D>,
            DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
        {
            _phantom: std::marker::PhantomData<(T, D)>,
        }

        impl<'de, T, D> serde::de::Visitor<'de> for SurfaceTessellationVisitor<T, D>
        where
            T: FloatingPoint + serde::Deserialize<'de>,
            D: DimName + DimNameSub<U1>,
            DefaultAllocator: Allocator<D>,
            DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
            <DefaultAllocator as nalgebra::allocator::Allocator<DimNameDiff<D, U1>>>::Buffer<T>:
                serde::Deserialize<'de>,
        {
            type Value = SurfaceTessellation<T, D>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct SurfaceTessellation")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
            where
                V: serde::de::MapAccess<'de>,
            {
                let mut points = None;
                let mut normals = None;
                let mut faces = None;
                let mut uvs = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Points => {
                            if points.is_some() {
                                return Err(serde::de::Error::duplicate_field("points"));
                            }
                            points = Some(map.next_value()?);
                        }
                        Field::Normals => {
                            if normals.is_some() {
                                return Err(serde::de::Error::duplicate_field("normals"));
                            }
                            normals = Some(map.next_value()?);
                        }
                        Field::Faces => {
                            if faces.is_some() {
                                return Err(serde::de::Error::duplicate_field("faces"));
                            }
                            faces = Some(map.next_value()?);
                        }
                        Field::Uvs => {
                            if uvs.is_some() {
                                return Err(serde::de::Error::duplicate_field("uvs"));
                            }
                            uvs = Some(map.next_value()?);
                        }
                    }
                }

                let points = points.ok_or_else(|| serde::de::Error::missing_field("points"))?;
                let normals = normals.ok_or_else(|| serde::de::Error::missing_field("normals"))?;
                let faces = faces.ok_or_else(|| serde::de::Error::missing_field("faces"))?;
                let uvs = uvs.ok_or_else(|| serde::de::Error::missing_field("uvs"))?;

                Ok(SurfaceTessellation {
                    points,
                    normals,
                    faces,
                    uvs,
                })
            }
        }

        const FIELDS: &[&str] = &["points", "normals", "faces", "uvs"];
        deserializer.deserialize_struct(
            "SurfaceTessellation",
            FIELDS,
            SurfaceTessellationVisitor {
                _phantom: std::marker::PhantomData,
            },
        )
    }
}

#[cfg(feature = "approx")]
impl<T, D> approx::AbsDiffEq for SurfaceTessellation<T, D>
where
    T: FloatingPoint + approx::AbsDiffEq<Epsilon = T>,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    type Epsilon = T;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        // Check faces first (exact match)
        if self.faces != other.faces {
            return false;
        }

        // Check lengths
        if self.points.len() != other.points.len()
            || self.normals.len() != other.normals.len()
            || self.uvs.len() != other.uvs.len()
        {
            return false;
        }

        // Check points
        for (p1, p2) in self.points.iter().zip(other.points.iter()) {
            if !p1
                .coords
                .iter()
                .zip(p2.coords.iter())
                .all(|(a, b)| approx::AbsDiffEq::abs_diff_eq(a, b, epsilon))
            {
                return false;
            }
        }

        // Check normals
        for (n1, n2) in self.normals.iter().zip(other.normals.iter()) {
            if !n1
                .iter()
                .zip(n2.iter())
                .all(|(a, b)| approx::AbsDiffEq::abs_diff_eq(a, b, epsilon))
            {
                return false;
            }
        }

        // Check uvs
        for (uv1, uv2) in self.uvs.iter().zip(other.uvs.iter()) {
            if !approx::AbsDiffEq::abs_diff_eq(&uv1.x, &uv2.x, epsilon)
                || !approx::AbsDiffEq::abs_diff_eq(&uv1.y, &uv2.y, epsilon)
            {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "approx")]
impl<T, D> approx::RelativeEq for SurfaceTessellation<T, D>
where
    T: FloatingPoint + approx::RelativeEq<Epsilon = T>,
    D: DimName + DimNameSub<U1>,
    DefaultAllocator: Allocator<D>,
    DefaultAllocator: Allocator<DimNameDiff<D, U1>>,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        // Check faces first (exact match)
        if self.faces != other.faces {
            return false;
        }

        // Check lengths
        if self.points.len() != other.points.len()
            || self.normals.len() != other.normals.len()
            || self.uvs.len() != other.uvs.len()
        {
            return false;
        }

        // Check points
        for (p1, p2) in self.points.iter().zip(other.points.iter()) {
            if !p1
                .coords
                .iter()
                .zip(p2.coords.iter())
                .all(|(a, b)| approx::RelativeEq::relative_eq(a, b, epsilon, max_relative))
            {
                return false;
            }
        }

        // Check normals
        for (n1, n2) in self.normals.iter().zip(other.normals.iter()) {
            if !n1
                .iter()
                .zip(n2.iter())
                .all(|(a, b)| approx::RelativeEq::relative_eq(a, b, epsilon, max_relative))
            {
                return false;
            }
        }

        // Check uvs
        for (uv1, uv2) in self.uvs.iter().zip(other.uvs.iter()) {
            if !approx::RelativeEq::relative_eq(&uv1.x, &uv2.x, epsilon, max_relative)
                || !approx::RelativeEq::relative_eq(&uv1.y, &uv2.y, epsilon, max_relative)
            {
                return false;
            }
        }

        true
    }
}
