use bevy::{
    color::palettes::css::{TOMATO, WHITE},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
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
) {
    let unit_circle =
        NurbsCurve2D::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap();
    // dbg!(unit_circle.try_length().unwrap(), 2.0 * std::f64::consts::PI);

    let samples = unit_circle.tessellate(Some(1e-8));
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

    let bbox = BoundingBox::from(&unit_circle);
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
