use bevy::{
    core::Zeroable,
    prelude::*,
    render::{camera::ScalingMode, mesh::VertexAttributeValues},
    window::close_on_esc,
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    plugin::PointsPlugin,
    prelude::{PointsMaterial, PointsMesh},
};
use nalgebra::{Point3, Vector3};

use curvo::prelude::*;

mod materials;
pub mod systems;

use materials::*;

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
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, (close_on_esc));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let add_curve =
        |curve: &NurbsCurve3D<f64>,
         commands: &mut Commands<'_, '_>,
         meshes: &mut ResMut<'_, Assets<Mesh>>,
         line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
         points_materials: &mut ResMut<'_, Assets<PointsMaterial>>| {
            // dbg!(curve.knots());
            let samples = curve.tessellate(Some(1e-12));
            let line_vertices = samples
                .iter()
                .map(|p| p.cast::<f32>())
                .map(|p| [p.x, p.y, p.z])
                .collect();
            let n = 1. / (samples.len() as f32);
            let line = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineStrip, default())
                .with_inserted_attribute(
                    Mesh::ATTRIBUTE_POSITION,
                    VertexAttributeValues::Float32x3(line_vertices),
                );
            /*
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_COLOR,
                VertexAttributeValues::Float32x4(samples.iter().enumerate().map(|(i, _)| {
                    let t = i as f32 * n;
                    Color::hsl(t * 360., 0.5, 0.5).rgba_to_vec4().into()
                }).collect()),
            );
            */
            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(line),
                    material: line_materials.add(LineMaterial {
                        color: Color::TOMATO,
                        ..Default::default()
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("curve"));

            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(PointsMesh {
                        vertices: curve
                            .dehomogenized_control_points()
                            .iter()
                            .map(|pt| pt.cast::<f32>().into())
                            .collect(),
                        colors: None,
                    }),
                    material: points_materials.add(PointsMaterial {
                        settings: bevy_points::material::PointsShaderSettings {
                            color: Color::ORANGE,
                            point_size: 0.05,
                            ..Default::default()
                        },
                        circle: true,
                        ..Default::default()
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("control points"));
        };

    /*
    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 0.0, 0.),
        Point3::new(-1.0, 0.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
    ];
        */

    let interpolation_target = vec![
        /*
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 0.0, 0.),
        Point3::new(-1.0, 0.0, 0.),
        */
        Point3::new(-1.0, -1.0, -1.),
        Point3::new(1.0, -1.0, -1.),
        Point3::new(3.0, -1.0, 1.),
        Point3::new(-4.0, 2.0, 1.),
        /*
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        */
    ];

    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(PointsMesh {
                vertices: interpolation_target
                    .iter()
                    .map(|pt| pt.cast::<f32>().into())
                    .collect(),
                colors: None,
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
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("interpolation targets"));

    let uniform =
        NurbsCurve3D::try_periodic_interpolate(&interpolation_target, KnotStyle::Chordal).unwrap();
    add_curve(
        &uniform,
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &mut points_materials,
    );

    /*
    let centripetal = NurbsCurve3D::try_periodic_interpolate(
        &interpolation_target
            .iter()
            .map(|p| (p.coords + Vector3::new(0., 3., 0.)).into())
            .collect::<Vec<_>>(),
        KnotStyle::Uniform,
    )
    .unwrap();
    add_curve(
        &centripetal,
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &mut points_materials,
    );
    */

    let scale = 5.;
    let orth = Camera3dBundle {
        projection: OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical(2.0),
            ..Default::default()
        }
        .into(),
        transform: Transform::from_translation(Vec3::new(0., 0., 3.)),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
}
