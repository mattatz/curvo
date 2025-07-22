use std::f64::consts::PI;

use bevy::prelude::*;
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use itertools::Itertools;
use materials::*;
use nalgebra::Point4;

use curvo::prelude::*;

use crate::misc::surface_2_mesh;

mod materials;
mod misc;

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
        app.add_systems(Startup, setup);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mesh_materials: ResMut<Assets<StandardMaterial>>,
) {
    // Create a simple bi-cubic surface with 2x2 patches
    let n = 6;
    let hn = n as f64 / 2.;
    let height = 0.5;
    let control_points = (0..=n)
        .map(|x| {
            let offset = x as f64 / n as f64;
            let ox = offset * PI;
            (0..=n)
                .map(|y| {
                    let oy = y as f64 / n as f64;
                    Point4::new(
                        x as f64 - hn,
                        (oy * PI + ox).cos() * height,
                        y as f64 - hn,
                        1.0,
                    )
                })
                .collect_vec()
        })
        .collect_vec();

    // Create knot vectors for a bi-cubic surface with internal knots
    let degree = 3;
    let u_knots = KnotVector::uniform(n - 1, degree);
    let v_knots = KnotVector::uniform(n - 1, degree);
    let surface = NurbsSurface3D::<f64>::new(degree, degree, u_knots, v_knots, control_points);

    let patches = surface.try_decompose().unwrap();
    let w = patches.len();

    for (ix, row) in patches.iter().enumerate() {
        let h = row.len();
        for (iy, patch) in row.iter().enumerate() {
            let mesh = surface_2_mesh(patch, None);
            commands.spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(mesh_materials.add(StandardMaterial {
                    base_color: Color::srgb(
                        ix as f32 / (w - 1) as f32,
                        iy as f32 / (h - 1) as f32,
                        0.,
                    ),
                    unlit: true,
                    cull_mode: None,
                    double_sided: true,
                    ..Default::default()
                })),
            ));
        }
    }

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        PanOrbitCamera::default(),
    ));
}
