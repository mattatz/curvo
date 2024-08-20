use std::{f64::consts::FRAC_PI_2, thread::spawn};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        view::screenshot::ScreenshotManager,
    },
    window::{close_on_esc, PrimaryWindow},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use nalgebra::{Point2, Point3, Rotation3, Translation3, Vector2, Vector3};

mod materials;
mod systems;

use curvo::prelude::*;
use systems::screenshot_on_spacebar;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (864., 480.).into(),
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
            .add_systems(Update, (screenshot_on_spacebar, close_on_esc));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let circle =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();

    let rectangle = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(0., 0.5),
        Point2::new(2., 0.5),
        Point2::new(2., -0.5),
        Point2::new(0., -0.5),
        Point2::new(0., 0.5),
    ]);

    let mut spawn_curve = |curve: &NurbsCurve2D<f64>| {
        let samples = curve.tessellate(Some(1e-8));
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, 0.])
            .collect();
        let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
        );
        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            ..Default::default()
        });
    };
    spawn_curve(&circle);
    spawn_curve(&rectangle);

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 0., 8.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
}
