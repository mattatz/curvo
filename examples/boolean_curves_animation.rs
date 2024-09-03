use std::f32::consts::PI;

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    window::close_on_esc,
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{Rotation2, Translation2};

mod boolean;
mod materials;
mod systems;

use curvo::prelude::*;
use operation::BooleanOperation;
use systems::screenshot_on_spacebar;

#[derive(Component)]
struct ProfileCurves(NurbsCurve2D<f64>, NurbsCurve2D<f64>);

#[derive(Component)]
struct BooleanMesh(BooleanOperation);

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
    line_materials: ResMut<Assets<LineMaterial>>,
    points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 0., 5.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
    commands.spawn(InfiniteGridBundle::default());

    let (subject, clip) = boolean::circle_rectangle_case();
    let (subject, clip) = boolean::periodic_interpolation_case();
    let (subject, clip) = boolean::island_case();
    commands.spawn((ProfileCurves(subject, clip),));

    let ops = [
        BooleanOperation::Intersection,
        BooleanOperation::Union,
        BooleanOperation::Difference,
    ];
    let n = ops.len();
    let inv_n = 1. / n as f32;
    let on = inv_n * 0.5;
    let h = n as f32 * 6.0;

    ops.into_iter().enumerate().for_each(|(i, op)| {
        let fi = i as f32 * inv_n + on - 0.5;
        let tr = Transform::from_xyz(fi * h, 0., 0.);
        commands
            .spawn(MaterialMeshBundle {
                mesh: meshes.add(Mesh::new(PrimitiveTopology::TriangleList, default())),
                material: normal_materials.add(NormalMaterial {
                    cull_mode: None,
                    ..Default::default()
                }),
                transform: tr,
                // visibility: Visibility::Hidden,
                ..Default::default()
            })
            .insert(BooleanMesh(op))
            .insert(Name::new(format!("{}", op)));
    });
}

fn boolean(
    mut gizmos: Gizmos,
    time: Res<Time>,
    profile: Query<&ProfileCurves>,
    booleans: Query<(&BooleanMesh, &Handle<Mesh>)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let delta = time.elapsed_seconds_f64() * 0.78;
    // let delta: f64 = 3.1309608852600004;
    // let delta: f64 = 3.1613637252600006;
    // let delta: f64 = 4.39098702276;
    println!("delta: {}", delta);

    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);

    if let Ok(profile) = profile.get_single() {
        let subject = &profile.0;
        let clip = profile.1.transformed(&trans.into());

        let tr = Transform::from_xyz(0., 5., 0.);

        [subject, &clip].iter().for_each(|curve| {
            let color = Color::rgba(1., 1., 1., 0.2);
            let pts = curve
                .tessellate(None)
                .iter()
                .map(|pt| pt.cast::<f32>())
                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                .collect_vec();
            gizmos.linestrip(pts, color);

            let pt = curve.point_at(curve.knots_domain().0);
            gizmos.sphere(
                pt.coords.cast::<f32>().to_homogeneous().into(),
                Quat::IDENTITY,
                1e-2,
                Color::WHITE,
            );
        });

        let option = CurveIntersectionSolverOptions {
            // minimum_distance: 1e-4,
            // knot_domain_division: 500,
            max_iters: 1000,
            ..Default::default()
        };

        booleans.iter().for_each(|(BooleanMesh(op), mesh)| {
            let regions = subject.boolean(*op, &clip, Some(option.clone()));
            if let Ok((regions, _)) = regions {
                let r = regions.first();
                let m = r.and_then(|r| r.tessellate(None).ok());
                if let Some(tess) = m {
                    if let Some(mesh) = meshes.get_mut(mesh) {
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_POSITION,
                            VertexAttributeValues::Float32x3(
                                tess.vertices()
                                    .iter()
                                    .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                                    .collect(),
                            ),
                        );
                        mesh.insert_attribute(
                            Mesh::ATTRIBUTE_NORMAL,
                            VertexAttributeValues::Float32x3(
                                tess.vertices().iter().map(|p| [0., 0., 1.]).collect(),
                            ),
                        );
                        mesh.insert_indices(Indices::U32(
                            tess.faces()
                                .iter()
                                .flat_map(|f| f.iter().map(|i| *i as u32))
                                .collect(),
                        ));
                    }
                }
            }
        });

        /*
        let intersections = subject
            .find_intersections(&other, Some(option.clone()))
            .unwrap();

        intersections.iter().for_each(|it| {
            let position = it.a().0.cast::<f32>().coords.to_homogeneous().into();
            gizmos.sphere(position, Quat::IDENTITY, 1e-1, Color::TOMATO);
        });
        */

        /*
        ops.into_iter().enumerate().for_each(|(i, op)| {
            let regions = subject.boolean(op, &clip, Some(option.clone()));
            match regions {
                Ok((regions, its)) => {
                    its.iter().for_each(|it| {
                        // gizmos.sphere(it.vertex().position().coords.cast::<f32>().to_homogeneous().into(), Quat::IDENTITY, 1e-2, Color::TOMATO);
                    });

                    let fi = i as f32 * inv_n + on - 0.5;
                    let tr = Transform::from_xyz(fi * h, 0., 0.);
                    regions.iter().for_each(|region| {
                        let pts = region
                            .exterior()
                            .tessellate(None)
                            .iter()
                            .map(|pt| pt.cast::<f32>())
                            .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                            .collect_vec();
                        gizmos.linestrip(pts, Color::TOMATO.with_a(0.5));
                        region.interiors().iter().for_each(|interior| {
                            let pts = interior
                                .tessellate(None)
                                .iter()
                                .map(|pt| pt.cast::<f32>())
                                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                                .collect_vec();
                            gizmos.linestrip(pts, Color::TURQUOISE.with_a(0.5));
                        });
                    });
                }
                Err(e) => {
                    println!("op: {}, error: {:?}, delta: {}", op, e, delta);
                }
            };
        });
        */
    }
}
