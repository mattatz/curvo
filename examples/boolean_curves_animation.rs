use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI};

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
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{
    Point2, Point3, Rotation2, Rotation3, Transform3, Translation2, Translation3, Vector2, Vector3,
};

mod materials;
mod systems;

use curvo::prelude::*;
use systems::screenshot_on_spacebar;

#[derive(Component)]
struct ProfileCurves(NurbsCurve2D<f64>, NurbsCurve2D<f64>);

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
            .add_systems(Update, (boolean, screenshot_on_spacebar, close_on_esc));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 0., 5.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));

    let dx = 2.25;
    let dy = 0.5;

    let source =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    let target = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(-dx, -dy),
        Point2::new(dx, -dy),
        Point2::new(dx, dy),
        Point2::new(-dx, dy),
        Point2::new(-dx, -dy),
    ]);

    /*
    let source =
        NurbsCurve2D::<f64>::try_ellipse(&Point2::origin(), &Vector2::x(), &(Vector2::y() * 1.5))
            .unwrap();
    let source = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &vec![
            Point2::new(-dx, -dy),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();

    let target = NurbsCurve2D::<f64>::try_periodic_interpolate(
        &vec![
            Point2::new(-dx, -dy),
            Point2::new(0., -dy * 0.5),
            Point2::new(dx, -dy),
            Point2::new(dx, dy),
            Point2::new(0., dy * 0.5),
            Point2::new(-dx, dy),
        ],
        3,
        KnotStyle::Centripetal,
    )
    .unwrap();
    */

    commands.spawn((ProfileCurves(source, target),));
}

fn boolean(mut gizmos: Gizmos, time: Res<Time>, profile: Query<&ProfileCurves>) {
    let delta = time.elapsed_seconds_f64() * 0.78;
    // let delta = 0_f64;
    // let delta: f64 = 5.5449795;
    println!("delta: {}", delta);

    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);

    if let Ok(profile) = profile.get_single() {
        let source = &profile.0;
        let other = profile.1.transformed(&trans.into());

        let tr = Transform::from_xyz(0., 5., 0.);

        [source, &other].iter().for_each(|curve| {
            let color = Color::rgba(1., 1., 1., 0.2);
            let pts = curve
                .tessellate(None)
                .iter()
                .map(|pt| pt.cast::<f32>())
                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                .collect_vec();
            gizmos.linestrip(pts, color);
        });

        let ops = [
            // BooleanOperation::Intersection,
            BooleanOperation::Union,
            // BooleanOperation::Difference,
        ];
        let n = ops.len();
        let inv_n = 1. / n as f32;
        let on = inv_n * 0.5;
        let h = n as f32 * 5.0;

        let option = CurveIntersectionSolverOptions {
            minimum_distance: 1e-5,
            knot_domain_division: 500,
            max_iters: 1000,
            ..Default::default()
        };

        ops.into_iter().enumerate().for_each(|(i, op)| {
            let regions = source.boolean(op, &other, Some(option.clone()));
            match regions {
                Ok((regions, its)) => {
                    let fi = i as f32 * inv_n + on - 0.5;
                    let tr = Transform::from_xyz(fi * h, 0., 0.);
                    regions.iter().for_each(|region| {
                        region.exterior().spans().iter().for_each(|curve| {
                            let pts = curve
                                .tessellate(None)
                                .iter()
                                .map(|pt| pt.cast::<f32>())
                                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                                .collect_vec();
                            gizmos.linestrip(pts, Color::TOMATO.with_a(0.5));
                        });
                        region.interiors().iter().for_each(|interior| {
                            interior.spans().iter().for_each(|curve| {});
                        });
                    });
                }
                Err(e) => {
                    println!("op: {}, error: {:?}, delta: {}", op, e, delta);
                }
            };
        });
    }
}
