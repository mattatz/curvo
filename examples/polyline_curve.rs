use bevy::{
    color::palettes::css::{TOMATO, WHITE},
    prelude::*,
    render::camera::ScalingMode,
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::{PointsMaterial, PointsShaderSettings},
    mesh::PointsMesh,
    plugin::PointsPlugin,
};
use misc::add_curve;
use nalgebra::Point2;

use curvo::prelude::*;

mod materials;
mod misc;

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
        app.add_systems(Startup, setup);
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
        .spawn((
            Mesh3d(
                meshes.add(PointsMesh {
                    vertices: points
                        .iter()
                        .map(|pt| pt.cast::<f32>())
                        .map(|pt| Vec3::new(pt.x, pt.y, 0.))
                        .collect(),
                    ..Default::default()
                }),
            ),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: PointsShaderSettings {
                    point_size: 0.05,
                    color: TOMATO.into(),
                    ..Default::default()
                },
                ..Default::default()
            })),
        ))
        .insert(Name::new("points"));

    let polyline_curve = NurbsCurve2D::polyline(&points, true);

    add_curve(
        &polyline_curve.elevate_dimension(),
        WHITE.into(),
        Some(1e-8),
        &mut commands,
        &mut meshes,
        &mut line_materials,
    );

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
        Transform::from_translation(Vec3::new(0., 0., 3.)).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
