use bevy::{
    color::palettes::css::{SEA_GREEN, TURQUOISE},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use boolean::CurveVariant;
use itertools::Itertools;
use materials::*;
use nalgebra::{Rotation2, Translation2, U2};

mod boolean;
mod materials;
mod systems;

use curvo::prelude::*;
use operation::BooleanOperation;
use systems::screenshot_on_spacebar;

#[derive(Component)]
struct ProfileCurves(CurveVariant, CurveVariant);

#[derive(Component)]
struct BooleanMesh(BooleanOperation);

const OPTION: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
    minimum_distance: 1e-4,
    knot_domain_division: 500,
    max_iters: 1000,
    // knot_domain_division: 100,
    // max_iters: 200,
    step_size_tolerance: 1e-8,
    cost_tolerance: 1e-10,
};

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
            .add_systems(Update, (boolean, screenshot_on_spacebar));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    _line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 5.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());

    // let (subject, clip) = boolean::circle_rectangle_case();
    // let (subject, clip) = boolean::periodic_interpolation_case();
    // let (subject, clip) = boolean::island_case();
    // let (subject, clip) = boolean::compound_circle_x_rectangle_case();
    // let (subject, clip) = boolean::rounded_rectangle_x_rectangle_case();
    // let (subject, clip) = boolean::rounded_t_shape_x_rectangle_case();
    // let (subject, clip) = boolean::rectangular_annulus_x_rectangle_case();
    let (subject, clip) = boolean::rectangular_annulus_x_t_shape_case();
    // let (subject, clip) = boolean::rectangular_annulus_x_rectangular_annulus_case();
    // let (clip, subject) = (subject, clip);
    commands.spawn((ProfileCurves(subject, clip),));

    let ops = [
        BooleanOperation::Intersection,
        BooleanOperation::Union,
        BooleanOperation::Difference,
    ];
    let n = ops.len();
    let inv_n = 1. / n as f32;
    let on = inv_n * 0.5;
    let h = n as f32 * 8.0;

    ops.into_iter().enumerate().for_each(|(i, op)| {
        let fi = i as f32 * inv_n + on - 0.5;
        let tr = Transform::from_xyz(fi * h, 0., 0.);
        commands
            .spawn((
                Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::TriangleList, default()))),
                MeshMaterial3d(normal_materials.add(NormalMaterial {
                    cull_mode: None,
                    ..Default::default()
                })),
                tr,
            ))
            .insert(BooleanMesh(op))
            .insert(Name::new(format!("{}", op)));
    });
}

fn boolean(
    mut gizmos: Gizmos,
    time: Res<Time>,
    profile: Query<&ProfileCurves>,
    booleans: Query<(&BooleanMesh, &Mesh3d, &Transform)>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let delta = time.elapsed_secs_f64() * 0.75;
    // let interval = 1e-1 * 2.;
    // let delta = (delta / interval).floor() * interval;
    // println!("delta: {}", delta);

    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    // let trans = Translation2::new(delta.cos(), 0.);

    if let Ok(profile) = profile.single() {
        let subject = &profile.0;
        let clip = profile.1.transformed(&trans.into());
        let tr = Transform::from_xyz(0., 5., 0.);

        [subject, &clip].iter().for_each(|curve| {
            let color = Color::srgba(1., 1., 1., 0.2);
            let pts = match curve {
                CurveVariant::Curve(c) => c.tessellate(None),
                CurveVariant::Compound(c) => c.tessellate(None),
                CurveVariant::Region(r) => r.exterior().tessellate(None),
            };
            let pts = pts
                .iter()
                .map(|pt| pt.cast::<f32>())
                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                .collect_vec();
            gizmos.linestrip(pts, color);

            if let CurveVariant::Region(r) = curve {
                r.interiors().iter().for_each(|interior| {
                    let pts = interior.tessellate(None);
                    let pts = pts
                        .iter()
                        .map(|pt| pt.cast::<f32>())
                        .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                        .collect_vec();
                    gizmos.linestrip(pts, color);
                });
            }
        });

        booleans.iter().for_each(|(BooleanMesh(op), mesh, trans)| {
            let regions = subject.boolean(*op, &clip, Some(OPTION.clone()));
            if let Ok(clip) = regions {
                let regions = clip.regions();
                let tess: PolygonMesh<f64, U2> =
                    regions.iter().filter_map(|r| r.tessellate(None).ok()).sum();

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
                            tess.vertices().iter().map(|_p| [0., 0., 1.]).collect(),
                        ),
                    );
                    mesh.insert_indices(Indices::U32(
                        tess.faces()
                            .iter()
                            .flat_map(|f| f.iter().map(|i| *i as u32))
                            .collect(),
                    ));
                }

                let trans = *trans * Transform::from_xyz(0., 0., 5.5);
                regions.iter().for_each(|region| {
                    let exterior = region.exterior().tessellate(None);
                    let pts = exterior
                        .iter()
                        .cycle()
                        .take(exterior.len() + 1)
                        .map(|pt| pt.cast::<f32>())
                        .map(|pt| trans * Vec3::new(pt.x, pt.y, 0.))
                        .collect_vec();
                    gizmos.linestrip(pts, SEA_GREEN.with_alpha(0.5));
                    region.interiors().iter().for_each(|interior| {
                        let interior = interior.tessellate(None);
                        let pts = interior
                            .iter()
                            .cycle()
                            .take(interior.len() + 1)
                            .map(|pt| pt.cast::<f32>())
                            .map(|pt| trans * Vec3::new(pt.x, pt.y, 0.))
                            .collect_vec();
                        gizmos.linestrip(pts, TURQUOISE.with_alpha(0.5));
                    });
                });
            }
        });

        /*
        let intersections = subject
            .find_intersection(&other, Some(option.clone()))
            .unwrap();

        intersections.iter().for_each(|it| {
            let position = it.a().0.cast::<f32>().coords.to_homogeneous().into();
            gizmos.sphere(position, Quat::IDENTITY, 1e-1, TOMATO);
        });
        */

        /*
        ops.into_iter().enumerate().for_each(|(i, op)| {
            let regions = subject.boolean(op, &clip, Some(option.clone()));
            match regions {
                Ok((regions, its)) => {
                    its.iter().for_each(|it| {
                        // gizmos.sphere(it.vertex().position().coords.cast::<f32>().to_homogeneous().into(), Quat::IDENTITY, 1e-2, TOMATO);
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
                        gizmos.linestrip(pts, TOMATO.with_a(0.5));
                        region.interiors().iter().for_each(|interior| {
                            let pts = interior
                                .tessellate(None)
                                .iter()
                                .map(|pt| pt.cast::<f32>())
                                .map(|pt| tr * Vec3::new(pt.x, pt.y, 0.))
                                .collect_vec();
                            gizmos.linestrip(pts, TURQUOISE.with_a(0.5));
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
