use std::f64::consts::TAU;

use bevy::{
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    },
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use itertools::Itertools;
use materials::*;
use nalgebra::{Point2, Point3, Vector2, Vector3};

use curvo::prelude::*;
mod materials;
mod misc;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
        .add_plugins(NormalMaterialPlugin)
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let surface = NurbsSurface::plane(
        Point3::new(2.5, 0., 5.),
        Vector3::x() * 2.5,
        Vector3::z() * 5.,
    );

    let trimmed_surface = TrimmedSurface::new(
        surface.clone(),
        Some(
            NurbsCurve2D::try_circle(&Point2::new(0.5, 0.5), &Vector2::x(), &Vector2::y(), 0.45)
                .unwrap()
                .into(),
        ),
        vec![
            NurbsCurve2D::try_circle(&Point2::new(0.5, 0.5), &Vector2::x(), &Vector2::y(), 0.2)
                .unwrap()
                .into(),
        ],
    );

    let option = AdaptiveTessellationOptions::<_> {
        norm_tolerance: 1e-3,
        ..Default::default()
    };

    let n = 10;
    let m = 20;
    let constraints = TrimmedSurfaceConstraints::new(
        Some(
            (0..n)
                .map(|i| i as f64 / (n - 1) as f64 * TAU)
                .collect_vec(),
        ),
        vec![Some(
            (0..m)
                .map(|i| i as f64 / (m - 1) as f64 * TAU)
                .collect_vec(),
        )],
    );
    let tess = trimmed_surface
        .constrained_tessellate(constraints, Some(option))
        .unwrap();
    let tess = tess.cast::<f32>();

    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect_vec();
    commands.spawn((
        Mesh3d(meshes.add(PointsMesh {
            vertices: vertices.clone(),
            ..Default::default()
        })),
        MeshMaterial3d(points_materials.add(PointsMaterial {
            settings: PointsShaderSettings {
                point_size: 0.02,
                color: Color::WHITE.into(),
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        })),
    ));

    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect_vec();

    let line_mesh = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        // triangle edges
        VertexAttributeValues::Float32x3(
            indices
                .chunks(3)
                .flat_map(|idx| {
                    [
                        vertices[idx[0] as usize].into(),
                        vertices[idx[1] as usize].into(),
                        vertices[idx[1] as usize].into(),
                        vertices[idx[2] as usize].into(),
                        vertices[idx[2] as usize].into(),
                        vertices[idx[0] as usize].into(),
                    ]
                })
                .collect_vec(),
        ),
    );
    commands.spawn((
        Mesh3d(meshes.add(line_mesh)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: Color::WHITE,
            ..Default::default()
        })),
    ));

    let tri_mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices.into_iter().map(|v| v.into()).collect()),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices));

    commands.spawn((
        Mesh3d(meshes.add(tri_mesh)),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            cull_mode: None,
            ..Default::default()
        })),
    ));

    let (umin, umax) = surface.u_knots_domain();
    let (vmin, vmax) = surface.v_knots_domain();
    let center = surface
        .point_at((umax + umin) * 0.5, (vmax + vmin) * 0.5)
        .cast::<f32>()
        .into();

    let scale = 5.;
    commands.spawn((
        Projection::Orthographic(OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 2.,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(center + Vec3::new(0., 0., 3.)).looking_at(center, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
