use std::f64::consts::TAU;

use bevy::{color::palettes::css::WHITE, prelude::*};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use misc::{add_curve, add_surface};
use nalgebra::Point3;

use curvo::prelude::*;
use rand::Rng;
use systems::screenshot_on_spacebar;
mod materials;
mod misc;
mod systems;

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
    let mut rng = rand::rng();
    let n = 12;
    let min_radius = 0.25;
    let max_radius = 2.5;
    let depth = 0.5;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.random::<f64>();
            let t = i as f64 / n as f64;
            let r = t * TAU;
            let rad = min_radius + g * (max_radius - min_radius);
            let x = r.cos() * rad;
            let y = r.sin() * rad;
            let z = depth * (rng.random::<f64>() - 0.5);
            Point3::new(x, y, z)
        })
        .collect();
    let profile = NurbsCurve3D::try_periodic(&points, 3).unwrap();

    let n = 32;
    let radius = 10.;
    let height = 20.;
    let angle = TAU * 2.0;
    let points: Vec<_> = (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            let r = t * angle;
            let x = r.cos() * radius;
            let z = r.sin() * radius;
            Point3::new(x, t * height, z)
        })
        .collect();
    let rail = NurbsCurve3D::interpolate(&points, 3).unwrap();
    add_curve(
        &rail,
        Some(WHITE.into()),
        Some(1e-3),
        &mut commands,
        &mut meshes,
        &mut line_materials,
    );

    let swept = NurbsSurface::try_sweep(&profile, &rail, Some(3)).unwrap();
    add_surface::<DefaultDivider>(
        &swept,
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        None,
    );

    commands.spawn((
        Transform::from_translation(Vec3::new(18., 18., 18.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
