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

    let circle =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();

    /*
    let rectangle = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(0., 0.5),
        Point2::new(2., 0.5),
        Point2::new(2., -0.5),
        Point2::new(0., -0.5),
        Point2::new(0., 0.5),
    ]);
    */

    let dx = 1.;
    let dy = 0.5;
    let rectangle = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(-dx, -dy),
        Point2::new(dx, -dy),
        Point2::new(dx, dy),
        Point2::new(-dx, dy),
        Point2::new(-dx, -dy),
    ]);

    commands.spawn((ProfileCurves(circle, rectangle),));
}

fn boolean(mut gizmos: Gizmos, time: Res<Time>, profile: Query<&ProfileCurves>) {
    let delta = time.elapsed_seconds_f64() * 0.78;
    // let delta = 0.;
    // let delta = FRAC_PI_2 - 1e-1 * 4.;
    // let delta = PI + FRAC_PI_2 ;
    // let delta = 3.30601908_f64;
    // println!("delta: {}", delta);

    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);

    if let Ok(profile) = profile.get_single() {
        let source = &profile.0;
        let other = profile.1.transformed(&trans.into());

        let tr = Transform::from_xyz(-4., 0., 0.);

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
            BooleanOperation::Union,
            BooleanOperation::Intersection,
            BooleanOperation::Difference,
        ];
        let n = ops.len();
        let inv_n = 1. / n as f32;
        let on = inv_n * 0.5;
        let h = n as f32 * 2.5;

        let option = CurveIntersectionSolverOptions {
            minimum_distance: 1e-2,
            knot_domain_division: 500,
            max_iters: 1000,
            ..Default::default()
        };

        /*
        let intersections = source.find_intersections(&other, Some(option.clone()));
        if let Ok(intersections) = intersections {
            let self_mid_parameter = (intersections[0].a().1 + intersections[1].a().1) / 2.;
            let mid = source.point_at(self_mid_parameter).cast::<f32>();
            gizmos.sphere(
                tr * Vec3::new(mid.x, mid.y, 0.),
                Quat::IDENTITY,
                1e-2,
                Color::BLUE,
            );

            let other_mid_parameter = (intersections[0].b().1 + intersections[1].b().1) / 2.;
            let mid2 = other.point_at(other_mid_parameter);
            let mid2_casted = mid2.cast::<f32>();
            gizmos.sphere(
                tr * Vec3::new(mid2_casted.x, mid2_casted.y, 0.),
                Quat::IDENTITY,
                1e-2,
                Color::ORANGE,
            );

            let ray = NurbsCurve::polyline(&vec![mid2.clone(), mid2 + Vector2::x() * 1e1]);
            let its = source.find_intersections(&ray, Some(option.clone()));
            if let Ok(its) = its {
                // println!("its: {:?}", its.len());
                its.iter()
                    .map(|it| {
                        let pt = it.a().0.cast::<f32>();
                        tr * Vec3::new(pt.x, pt.y, 0.)
                    })
                    .for_each(|pt| {
                        gizmos.sphere(pt, Quat::IDENTITY, 1e-2 * 3., Color::PINK);
                    });
            }

            intersections
                .iter()
                .map(|it| {
                    let pt = it.a().0.cast::<f32>();
                    tr * Vec3::new(pt.x, pt.y, 0.)
                })
                .for_each(|pt| {
                    gizmos.sphere(pt, Quat::IDENTITY, 1e-2, Color::RED);
                });

            intersections
                .iter()
                .map(|it| {
                    let pt = it.b().0.cast::<f32>();
                    tr * Vec3::new(pt.x, pt.y, 0.)
                })
                .for_each(|pt| {
                    gizmos.sphere(pt, Quat::IDENTITY, 1e-2, Color::GREEN);
                });
        }
        */

        ops.into_iter().enumerate().for_each(|(i, op)| {
            let regions = source.boolean(op, &other, Some(option.clone()));
            if let Ok(regions) = regions {
                let fi = i as f32 * inv_n + on - 0.5;
                let tr = Transform::from_xyz(0., fi * h, 0.);
                regions.iter().for_each(|region| {
                    region.exterior().spans().iter().for_each(|curve| {
                        let pts = curve
                            .tessellate(None)
                            .iter()
                            .map(|pt| pt.cast::<f32>())
                            .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                            .collect_vec();
                        gizmos.linestrip(pts, Color::TOMATO);
                    });
                    region.interiors().iter().for_each(|interior| {
                        interior.spans().iter().for_each(|curve| {});
                    });
                });
            }
        });
    }
}
