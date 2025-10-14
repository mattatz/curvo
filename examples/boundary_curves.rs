use std::f64::consts::FRAC_PI_2;

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

mod materials;
mod systems;

use curvo::prelude::*;
use systems::screenshot_on_spacebar;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (640., 480.).into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
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
        app.add_systems(Startup, setup)
            .add_systems(Update, screenshot_on_spacebar);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 2.0, 0.),
        Point3::new(1.0, 2.5, 0.),
    ];
    let interpolated = NurbsCurve3D::<f64>::interpolate(&interpolation_target, 3).unwrap();

    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let translation = Translation3::new(0., 0., 1.5);
    let m = translation * rotation;
    let front = interpolated.transformed(&(translation.inverse()).into());
    let back = interpolated.transformed(&m.into());
    let lofted = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();

    let option = AdaptiveTessellationOptions::<_>::default().with_norm_tolerance(1e-2);
    let tess = lofted.tessellate(Some(option));
    let tess = tess.cast::<f32>();
    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();

    let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices));

    commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(normal_materials.add(NormalMaterial {
                opacity: 0.05,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            })),
            Visibility::Hidden,
        ))
        .insert(Name::new("lofted"));

    let boundary = lofted.try_boundary_curves();
    if let Ok(boundary) = boundary {
        boundary.iter().for_each(|curve| {
            let (a, b) = curve.knots_domain();
            let samples = curve.sample_regular_range(a, b, 128);
            let line_vertices: Vec<_> = samples
                .iter()
                .map(|p| p.cast::<f32>())
                .map(|p| p.into())
                .collect();
            let n = line_vertices.len();
            let line = Mesh::new(PrimitiveTopology::LineStrip, default())
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    VertexAttributeValues::Float32x3(line_vertices),
                )
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_COLOR,
                    VertexAttributeValues::Float32x4(
                        (0..n)
                            .map(|i| Color::hsl(((i as f32) / n as f32) * 100., 0.5, 0.5))
                            .map(|c| c.to_srgba().to_f32_array())
                            .collect(),
                    ),
                );
            commands
                .spawn((
                    Mesh3d(meshes.add(line)),
                    MeshMaterial3d(line_materials.add(LineMaterial {
                        color: Color::WHITE,
                        ..Default::default()
                    })),
                ))
                .insert(Name::new("boundary curve"));
        });
    }

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 3., 8.)),
        PanOrbitCamera::default(),
    ));
}
