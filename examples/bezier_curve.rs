use bevy::{
    color::palettes::css::{TOMATO, WHITE},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use nalgebra::{Point2, Vector2};

use curvo::prelude::*;

mod materials;

use materials::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
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
) {
    let points = [
        Point2::new(-1.0, -1.0),
        Point2::new(0.5, -1.0),
        Point2::new(1.0, -0.5),
        Point2::new(1.0, 1.0),
    ];

    let bezier = NurbsCurve2D::bezier(&points);

    commands
        .spawn((
            Mesh3d(
                meshes.add(PointsMesh {
                    vertices: points
                        .iter()
                        .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                        .collect(),
                    ..Default::default()
                }),
            ),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: WHITE.into(),
                    point_size: 0.05,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            })),
        ))
        .insert(Name::new("control points"));

    let samples = bezier.tessellate(Some(1e-8));
    let line_vertices = samples
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, 0.])
        .collect();
    let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
    commands
        .spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: TOMATO.into(),
                ..Default::default()
            })),
        ))
        .insert(Name::new("curve"));

    let bbox = BoundingBox::from(&bezier);
    let vertices = bbox
        .lines()
        .iter()
        .flat_map(|(a, b)| [a, b].iter().map(|p| [p.x, p.y, 0.]).collect::<Vec<_>>())
        .collect();
    let line = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    commands
        .spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: WHITE.into(),
                ..Default::default()
            })),
        ))
        .insert(Name::new("bounding box"));

    let scale = 5.;
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 1.,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(Vec3::new(0., 0., 3.)).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
