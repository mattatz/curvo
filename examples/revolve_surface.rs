use std::f64::consts::TAU;

use bevy::{color::palettes::css::WHITE, prelude::*};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use misc::{add_curve, add_surface};
use nalgebra::{Point3, Vector3};

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
    let n = 10;
    let depth = 8.;
    let min_radius = 1.0;
    let max_radius = 2.5;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.random::<f64>();
            let t = i as f64 / n as f64;
            let rad = min_radius + g * (max_radius - min_radius);
            let r = 0_f64;
            let x = r.cos() * rad;
            let z = r.sin() * rad;
            let y = depth * t;
            Point3::new(x, y, z)
        })
        .collect();
    let profile = NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    add_curve(
        &profile,
        Some(WHITE.into()),
        Some(1e-3),
        &mut commands,
        &mut meshes,
        &mut line_materials,
    );

    let revolved =
        NurbsSurface::try_revolve(&profile, &Point3::origin(), &Vector3::y(), TAU / 4.0 * 3.0)
            .unwrap();
    add_surface::<DefaultDivider>(
        &revolved,
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
