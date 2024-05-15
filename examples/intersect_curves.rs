use bevy::{
    core::Zeroable,
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
    window::close_on_esc,
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    plugin::PointsPlugin,
    prelude::{PointsMaterial, PointsMesh},
};
use nalgebra::Point3;

use curvo::prelude::*;

mod materials;

use materials::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
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
    let points = vec![
        Point3::new(-1.0, -1.0, -0.5),
        Point3::new(1.0, -1.0, 0.2),
        Point3::new(1.0, 1.0, 0.0),
        Point3::new(-1.0, 1.0, -0.2),
        Point3::new(-1.0, 2.0, 0.0),
        Point3::new(1.0, 2.5, 0.5),
    ];
    let curve = NurbsCurve3D::try_interpolate(&points, 3).unwrap();
    let line_vertices = curve
        .tessellate(Some(1e-8))
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, p.z])
        .collect();
    let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
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

    let mirror = points
        .iter()
        .map(|pt| Point3::new(-pt.x, pt.y, pt.z))
        .collect::<Vec<_>>();
    let mirrored = NurbsCurve3D::try_interpolate(&mirror, curve.degree()).unwrap();
    let line_vertices = mirrored
        .tessellate(Some(1e-8))
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, p.z])
        .collect();
    let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: Color::LIME_GREEN,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("mirrored"));

    let traversed = BoundingBoxTraversal::try_traverse(&curve, &mirrored, None);
    if let Ok(traversed) = traversed {
        traversed.pairs_iter().for_each(|(a, b)| {
            let b0 = a.bounding_box();
            let b1 = b.bounding_box();
            let vertices = b0
                .lines()
                .iter()
                .chain(b1.lines().iter())
                .flat_map(|(a, b)| [a, b].iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>())
                .collect();
            let line = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(vertices),
            );
            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(line),
                    material: line_materials.add(LineMaterial {
                        color: Color::WHITE,
                        ..Default::default()
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("bounding box"));
        });
    }

    let center = Vec3::ZERO;
    let orth = Camera3dBundle {
        transform: Transform::from_translation(center + Vec3::new(0., 0., 8.))
            .looking_at(center, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            x_axis_color: Color::BLACK,
            z_axis_color: Color::BLACK,
            ..Default::default()
        },
        ..Default::default()
    });
}
