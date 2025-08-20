use bevy::{
    color::palettes::css::YELLOW,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_egui::EguiContexts;
use bevy_normal_material::prelude::NormalMaterial;
use curvo::prelude::{
    AdaptiveTessellationOptions, CompoundCurve3D, NurbsCurve3D, NurbsSurface3D,
    SurfaceTessellation3D, Tessellation,
};

use crate::LineMaterial;

pub enum CurveVariant<'a> {
    NurbsCurve(&'a NurbsCurve3D<f64>),
    CompoundCurve(&'a CompoundCurve3D<f64>),
}

impl<'a> From<&'a NurbsCurve3D<f64>> for CurveVariant<'a> {
    fn from(curve: &'a NurbsCurve3D<f64>) -> Self {
        CurveVariant::NurbsCurve(curve)
    }
}

impl<'a> From<&'a CompoundCurve3D<f64>> for CurveVariant<'a> {
    fn from(curve: &'a CompoundCurve3D<f64>) -> Self {
        CurveVariant::CompoundCurve(curve)
    }
}

#[allow(unused)]
pub fn add_curve<'a, C: Into<CurveVariant<'a>>>(
    curve: C,
    color: Option<Color>,
    tolerance: Option<f64>,
    commands: &mut Commands<'_, '_>,
    meshes: &mut ResMut<'_, Assets<Mesh>>,
    line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
) {
    let curve: CurveVariant<'a> = curve.into();

    let samples = match curve {
        CurveVariant::NurbsCurve(n) => n.tessellate(tolerance),
        CurveVariant::CompoundCurve(c) => c
            .spans()
            .iter()
            .flat_map(|span| span.tessellate(tolerance))
            .collect(),
    };

    let line_vertices: Vec<_> = samples
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| p.into())
        .collect();
    let n = line_vertices.len();
    let mut line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
    if color.is_none() {
        line.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(
                (0..n)
                    .map(|i| Color::hsl(((i as f32) / n as f32) * 300., 0.5, 0.5))
                    .map(|c| c.to_srgba().to_f32_array())
                    .collect(),
            ),
        );
    }
    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: color.unwrap_or(Color::WHITE),
            ..Default::default()
        })),
    ));
}

#[allow(unused)]
pub fn add_regular_curve<'a, C: Into<CurveVariant<'a>>>(
    curve: C,
    color: Option<Color>,
    divs: usize,
    commands: &mut Commands<'_, '_>,
    meshes: &mut ResMut<'_, Assets<Mesh>>,
    line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
) {
    let curve: CurveVariant<'a> = curve.into();

    let samples = match curve {
        CurveVariant::NurbsCurve(n) => {
            n.sample_regular_range(n.knots_domain().0, n.knots_domain().1, divs)
        }
        CurveVariant::CompoundCurve(c) => c
            .spans()
            .iter()
            .flat_map(|span| {
                span.sample_regular_range(span.knots_domain().0, span.knots_domain().1, divs)
            })
            .collect(),
    };

    let line_vertices: Vec<_> = samples
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| p.into())
        .collect();
    let n = line_vertices.len();
    let mut line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
    if color.is_none() {
        line.insert_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(
                (0..n)
                    .map(|i| Color::hsl(((i as f32) / n as f32) * 300., 0.5, 0.5))
                    .map(|c| c.to_srgba().to_f32_array())
                    .collect(),
            ),
        );
    }
    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: color.unwrap_or(Color::WHITE),
            ..Default::default()
        })),
    ));
}

#[allow(unused)]
pub fn surface_2_mesh(
    surface: &NurbsSurface3D<f64>,
    option: Option<AdaptiveTessellationOptions<f64>>,
) -> Mesh {
    let option = option.unwrap_or_default();
    let tess = surface.tessellate(Some(option));
    let tess = tess.cast::<f32>();

    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();

    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices))
}

#[allow(unused)]
pub fn surface_2_regular_mesh(surface: &NurbsSurface3D<f64>, divs_u: usize, divs_v: usize) -> Mesh {
    let tess = surface.regular_tessellate(divs_u, divs_v);
    let tess = tess.cast::<f32>();

    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();

    Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices))
}

#[allow(unused)]
pub fn add_surface(
    surface: &NurbsSurface3D<f64>,
    commands: &mut Commands<'_, '_>,
    meshes: &mut ResMut<'_, Assets<Mesh>>,
    normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
    option: Option<AdaptiveTessellationOptions<f64>>,
) {
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());

    let option = option.unwrap_or_default();
    let tess = surface.tessellate(Some(option));
    let tess = tess.cast::<f32>();

    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();

    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs));
    mesh.insert_indices(Indices::U32(indices));

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            cull_mode: None,
            ..Default::default()
        })),
    ));
}

#[allow(unused)]
pub fn add_surface_normals(
    tess: &SurfaceTessellation3D<f32>,
    commands: &mut Commands<'_, '_>,
    meshes: &mut ResMut<'_, Assets<Mesh>>,
    line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
) {
    let mut line_list = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineList, default());
    let normal_length = 0.15;
    let normals = tess.normals();

    let vertices = tess
        .points()
        .iter()
        .enumerate()
        .flat_map(|(i, p)| {
            let pt: Vec3 = (*p).into();
            let normal: Vec3 = normals[i].normalize().into();
            [pt, pt + normal * normal_length]
        })
        .map(|p| p.to_array())
        .collect();

    line_list.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );

    commands
        .spawn((
            Mesh3d(meshes.add(line_list)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: YELLOW.into(),
                ..Default::default()
            })),
        ))
        .insert(Name::new("normal"));
}

#[allow(unused)]
pub fn absorb_egui_inputs(mut mouse: ResMut<ButtonInput<MouseButton>>, mut contexts: EguiContexts) {
    if contexts.ctx_mut().is_pointer_over_area() {
        mouse.reset_all();
    }
}
