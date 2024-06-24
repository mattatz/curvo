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
use bevy_points::{
    material::{PointsMaterial, PointsShaderSettings},
    mesh::PointsMesh,
    plugin::PointsPlugin,
};
use nalgebra::Point2;

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
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let points = vec![
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 0.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-1.0, 1.0),
        Point2::new(1.0, 1.0),
    ];
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(PointsMesh {
                vertices: points
                    .iter()
                    .map(|pt| pt.cast::<f32>())
                    .map(|pt| Vec3::new(pt.x, pt.y, 0.))
                    .collect(),
                ..Default::default()
            }),
            material: points_materials.add(PointsMaterial {
                settings: PointsShaderSettings {
                    point_size: 0.05,
                    color: Color::TOMATO,
                    ..Default::default()
                },
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("points"));

    let polyline_curve = NurbsCurve2D::polyline(&points);

    let samples = polyline_curve.tessellate(Some(1e-8));
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
                color: Color::WHITE,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("curve"));

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
