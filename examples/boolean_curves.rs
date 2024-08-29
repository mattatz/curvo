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
use operation::BooleanOperation;
use status::Status;
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

    /*
    let subject =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();
    let clip = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(-dx, -dy),
        Point2::new(dx, -dy),
        Point2::new(dx, dy),
        Point2::new(-dx, dy),
        Point2::new(-dx, -dy),
    ]);
    */

    let subject = NurbsCurve2D::<f64>::try_periodic_interpolate(
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

    let clip = NurbsCurve2D::<f64>::try_periodic_interpolate(
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

    // let delta: f64 = 0.;
    let delta: f64 = 0.9066810530447318;
    // let delta: f64 = 1.;
    // let delta: f64 = 5.53018797552;
    // let delta: f64 = 1.1112463049999999;
    let delta: f64 = 5.52497952474;
    let delta: f64 = 5.39335530474;
    // let delta: f64 = 6.491389612500001;
    // let delta: f64 = 0.59434638198;
    let delta: f64 = 7.372849087500001;
    let delta: f64 = 4.8583862605200006;
    let delta: f64 = 2.6065769597400004;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = clip.transformed(&trans.into());

    let spawn_curve = |commands: &mut Commands,
                       meshes: &mut ResMut<Assets<Mesh>>,
                       line_materials: &mut ResMut<Assets<LineMaterial>>,
                       transform: Transform,
                       curve: &NurbsCurve2D<f64>,
                       color: Option<Color>| {
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
                color: color.unwrap_or(Color::WHITE.with_a(0.25)),
                opacity: 0.6,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            }),
            transform,
            ..Default::default()
        });
    };

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        Transform::default(),
        &subject,
        None,
    );
    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        Transform::default(),
        &clip,
        None,
    );

    let option = CurveIntersectionSolverOptions {
        minimum_distance: 1e-4,
        // knot_domain_division: 500,
        // max_iters: 1000,
        ..Default::default()
    };

    let intersections = subject
        .find_intersections(&clip, Some(option.clone()))
        .unwrap();
    // println!("intersections: {:?}", intersections.len());

    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(PointsMesh {
            vertices: intersections
                .iter()
                .map(|it| it.a().0.cast::<f32>().to_homogeneous().into())
                .collect(),
            colors: Some(
                intersections
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let hue = i as f32 / intersections.len() as f32;
                        Color::hsl(hue * 360., 0.5, 0.5)
                    })
                    .collect(),
            ),
        }),
        material: points_materials.add(PointsMaterial {
            settings: bevy_points::material::PointsShaderSettings {
                color: Color::WHITE,
                point_size: 0.05,
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        }),
        ..Default::default()
    });

    let parameter_eps: f64 = 1e-4;
    intersections.iter().for_each(|it| {
        let a0 = subject.point_at(it.a().1 - parameter_eps);
        let a1 = subject.point_at(it.a().1 + parameter_eps);
        let b0 = clip.point_at(it.b().1 - parameter_eps);
        let b1 = clip.point_at(it.b().1 + parameter_eps);
        let lines = [Line::new(a0, a1), Line::new(b0, b1)];
        lines.iter().enumerate().for_each(|(i, line)| {
            let mesh = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(vec![
                    line.start().coords.cast::<f32>().to_homogeneous().into(),
                    line.end().coords.cast::<f32>().to_homogeneous().into(),
                ]),
            );
            commands.spawn(MaterialMeshBundle {
                mesh: meshes.add(mesh),
                material: line_materials.add(LineMaterial {
                    color: if i == 0 { Color::RED } else { Color::BLUE },
                    opacity: 1.0,
                    alpha_mode: AlphaMode::Blend,
                    ..Default::default()
                }),
                ..Default::default()
            });
        });
    });

    let ops = [
        // BooleanOperation::Union,
        // BooleanOperation::Intersection,
        BooleanOperation::Difference,
    ];
    let n = ops.len();
    let inv_n = 1. / n as f32;
    let on = inv_n * 0.5;
    let h = n as f32 * 2.5;

    ops.into_iter().enumerate().for_each(|(i, op)| {
        let (regions, intersections) = subject.boolean(op, &clip, Some(option.clone())).unwrap();
        let fi = i as f32 * inv_n + on - 0.5;
        let tr = Transform::from_xyz(fi * h, 0., 0.);

        intersections.iter().for_each(|it| {
            // println!("intersection: {:?}", it);
        });

        let points = intersections
            .iter()
            .enumerate()
            .map(|(i, it)| {
                let pt = it.vertex().position().cast::<f32>();
                tr * Vec3::new(pt.x, pt.y, i as f32 * 1e-1)
            })
            .collect();

        let colors = intersections
            .iter()
            .map(|it| match it.status() {
                Status::None => Color::WHITE,
                Status::Enter => Color::BLUE,
                Status::Exit => Color::RED,
            })
            .collect();

        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(PointsMesh {
                vertices: points,
                colors: Some(colors),
            }),
            material: points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: Color::WHITE,
                    point_size: 0.05,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            }),
            ..Default::default()
        });

        regions.iter().for_each(|region| {
            region.exterior().spans().iter().for_each(|curve| {
                spawn_curve(
                    &mut commands,
                    &mut meshes,
                    &mut line_materials,
                    tr,
                    curve,
                    Some(Color::GREEN),
                );
            });
            region.interiors().iter().for_each(|interior| {
                interior.spans().iter().for_each(|curve| {
                    spawn_curve(
                        &mut commands,
                        &mut meshes,
                        &mut line_materials,
                        tr,
                        curve,
                        Some(Color::RED),
                    );
                });
            });
        });
    });
}
