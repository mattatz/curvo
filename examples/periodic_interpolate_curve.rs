use bevy::{
    color::palettes::{
        css::{BLUE, TOMATO, WHITE},
        tailwind::LIME_300,
    },
    prelude::*,
    render::{camera::ScalingMode, mesh::VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    plugin::PointsPlugin,
    prelude::{PointsMaterial, PointsMesh},
};
use nalgebra::Point3;

use curvo::prelude::*;

mod materials;
mod systems;

use materials::*;
use rand_distr::{Distribution, UnitSphere};
use systems::screenshot_on_spacebar;

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
            .add_systems(Update, screenshot_on_spacebar);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let add_curve = |curve: &NurbsCurve3D<f64>,
                     commands: &mut Commands<'_, '_>,
                     meshes: &mut ResMut<'_, Assets<Mesh>>,
                     line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
                     points_materials: &mut ResMut<'_, Assets<PointsMaterial>>,
                     color: Color| {
        // dbg!(curve.knots());
        let samples = curve.tessellate(Some(1e-12));
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, p.z])
            .collect();
        let _n = 1. / (samples.len() as f32);
        let line = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineStrip, default())
            .with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(line_vertices),
            );
        /*
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_COLOR,
            VertexAttributeValues::Float32x4(
                samples
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let t = i as f32 * n;
                        Color::hsl(t * 360., 0.5, 0.5).rgba_to_vec4().into()
                    })
                    .collect(),
            ),
        );
        */

        commands
            .spawn(MaterialMeshBundle {
                mesh: meshes.add(line),
                material: line_materials.add(LineMaterial {
                    color,
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
                        color: color.into(),
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

    let interpolation_target: Vec<Point3<f64>> = (0..8)
        .map(|_i| {
            let sample: [f64; 3] = UnitSphere.sample(&mut rand::thread_rng());
            Point3::from_slice(&sample) * 5.
        })
        .collect();

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
                    color: WHITE.into(),
                    point_size: 0.072,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("interpolation targets"));

    let degree = 3;

    [
        (KnotStyle::Uniform, TOMATO),
        (KnotStyle::Chordal, BLUE),
        (KnotStyle::Centripetal, LIME_300),
    ]
    .into_iter()
    .for_each(|(knot, color)| {
        let curve =
            NurbsCurve3D::try_periodic_interpolate(&interpolation_target, degree, knot).unwrap();
        add_curve(
            &curve,
            &mut commands,
            &mut meshes,
            &mut line_materials,
            &mut points_materials,
            color.into(),
        );
    });

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
