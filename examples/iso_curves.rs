use std::f64::consts::FRAC_PI_2;

use bevy::{color::palettes::css::TOMATO, prelude::*};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use materials::*;
use misc::*;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};
use systems::*;

use curvo::prelude::*;
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
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    _normal_materials: ResMut<'_, Assets<NormalMaterial>>,
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

    let direction = UVDirection::V;
    let (min, max) = match direction {
        UVDirection::U => lofted.u_knots_domain(),
        UVDirection::V => lofted.v_knots_domain(),
    };
    let div = 64;
    for i in 0..div {
        let u = min + (max - min) * i as f64 / (div - 1) as f64;
        // println!("u:{}, min:{}, max:{}", u, min, max);
        let iso = lofted.try_isocurve(u, direction);
        match iso {
            Ok(curve) => {
                add_curve(
                    &curve,
                    None,
                    Some(1e-8),
                    &mut commands,
                    &mut meshes,
                    &mut line_materials,
                );

                let (start, end) = curve.knots_domain();
                let points = curve.sample_regular_range(start, end, 16);
                commands
                    .spawn((
                        Mesh3d(meshes.add(PointsMesh {
                            vertices: points.iter().map(|pt| pt.cast::<f32>().into()).collect(),
                            ..Default::default()
                        })),
                        MeshMaterial3d(points_materials.add(PointsMaterial {
                            settings: PointsShaderSettings {
                                point_size: 0.05,
                                color: TOMATO.into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        })),
                        Visibility::Hidden,
                    ))
                    .insert(Name::new("points"));
            }
            Err(e) => {
                println!("error:{}", e);
            }
        }
    }

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 3., 8.)),
        PanOrbitCamera::default(),
    ));
}
