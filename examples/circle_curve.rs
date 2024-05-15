use bevy::{
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
    window::close_on_esc,
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use nalgebra::Point3;

use curvo::prelude::*;

mod materials;

use materials::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, close_on_esc);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    let corner_weight = 1. / 2.;
    let unit_circle = NurbsCurve2D::try_new(
        2,
        vec![
            Point3::new(1.0, 0.0, 1.),
            Point3::new(1.0, 1.0, 1.0) * corner_weight,
            Point3::new(-1.0, 1.0, 1.0) * corner_weight,
            Point3::new(-1.0, 0.0, 1.),
            Point3::new(-1.0, -1.0, 1.0) * corner_weight,
            Point3::new(1.0, -1.0, 1.0) * corner_weight,
            Point3::new(1.0, 0.0, 1.),
        ],
        vec![0., 0., 0., 1. / 4., 1. / 2., 1. / 2., 3. / 4., 1., 1., 1.],
    )
    .unwrap();
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
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: Color::TOMATO,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
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
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("bounding box"));

    let scale = 5.;
    let orth = Camera3dBundle {
        projection: OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical(2.0),
            ..Default::default()
        }
        .into(),
        transform: Transform::from_translation(Vec3::new(0., 0., 3.))
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
}
