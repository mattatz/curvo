use std::f64::consts::TAU;

use bevy::{
    color::palettes::css::WHITE,
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{material::PointsMaterial, mesh::PointsMesh, plugin::PointsPlugin};
use materials::*;
use nalgebra::Point3;

use curvo::prelude::*;
use rand::Rng;
mod materials;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
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
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let mut rng = rand::rng();
    let n = 12;
    let min_radius = 0.25;
    let max_radius = 2.5;
    let depth = 10.;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.random::<f64>();
            let t = i as f64 / n as f64;
            let r = t * TAU;
            let rad = min_radius + g * (max_radius - min_radius);
            let x = r.cos() * rad;
            let z = r.sin() * rad;
            let y = depth * t;
            Point3::new(x, y, z)
        })
        .collect();
    let curve = NurbsCurve3D::try_interpolate(&points, 3).unwrap();

    let vertices = curve
        .cast::<f32>()
        .tessellate(Some(1e-3))
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();
    let mesh = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: Color::WHITE,
                ..Default::default()
            })),
        ))
        .insert(Name::new("curve"));

    let segments = curve.try_decompose_bezier_segments().unwrap();
    let n = segments.len();
    segments.iter().enumerate().for_each(|(i, segment)| {
        let segment = segment.cast::<f32>();
        let vertices = segment
            .tessellate(Some(1e-6))
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        let mesh = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        );
        let t = i as f32 / n as f32;
        commands
            .spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color: Color::hsl(t * 360., 0.5, 0.5),
                    ..Default::default()
                })),
            ))
            .insert(Name::new("bezier segment"));

        let (start, end) = segment.knots_domain();
        let p0 = segment.point_at(start);
        let p1 = segment.point_at(end);

        commands
            .spawn((
                Mesh3d(meshes.add(PointsMesh {
                    vertices: [p0, p1].iter().map(|pt| (*pt).into()).collect(),
                    colors: None,
                })),
                MeshMaterial3d(points_materials.add(PointsMaterial {
                    settings: bevy_points::material::PointsShaderSettings {
                        color: WHITE.into(),
                        point_size: 0.05,
                        ..Default::default()
                    },
                    circle: true,
                    ..Default::default()
                })),
            ))
            .insert(Name::new("interpolation targets"));
    });

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
