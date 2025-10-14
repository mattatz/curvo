use std::f64::consts::TAU;

use bevy::{
    color::palettes::css::{GREEN, RED, WHITE, YELLOW},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use materials::*;
use nalgebra::{Matrix4, Point3};

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
    let curve = NurbsCurve3D::interpolate(&points, 3).unwrap();

    let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, default());
    let vertices = curve
        .cast::<f32>()
        .tessellate(Some(1e-6))
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: WHITE.into(),
                ..Default::default()
            })),
        ))
        .insert(Name::new("curve"));

    let (start, end) = curve.knots_domain();
    let samples = 32;
    let span = (end - start) / ((samples - 1) as f64);
    let parameters: Vec<_> = (0..samples).map(|i| start + span * (i as f64)).collect();
    let frames = curve
        .compute_frenet_frames(&parameters)
        .into_iter()
        .map(|f| f.cast::<f32>());

    let size = 0.25;
    let hs = size * 0.5;
    let vertices = vec![
        [-hs, -hs, 0.],
        [hs, -hs, 0.],
        [hs, hs, 0.],
        [-hs, hs, 0.],
        [-hs, -hs, 0.],
    ];

    let mut tangents = vec![];
    let mut normals = vec![];
    let mut binormals = vec![];

    let length = 0.15;
    frames.enumerate().for_each(|(i, frame)| {
        let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, default());
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices.clone()),
        );

        let matrix: Matrix4<f32> = frame.matrix().into();
        let tr = Transform::from_matrix(matrix.into());

        commands
            .spawn((
                Mesh3d(meshes.add(mesh)),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color: WHITE.into(),
                    ..Default::default()
                })),
                tr,
            ))
            .insert(Name::new(format!("frame_{}", i)));

        let p: Vec3 = (*frame.position()).into();
        let t: Vec3 = (*frame.tangent()).into();
        let n: Vec3 = (*frame.normal()).into();
        let b: Vec3 = (*frame.binormal()).into();
        tangents.push(p);
        tangents.push(p + t * length);
        normals.push(p);
        normals.push(p + n * length);
        binormals.push(p);
        binormals.push(p + b * length);
    });

    let add_arrows = |commands: &mut Commands<'_, '_>,
                      meshes: &mut ResMut<'_, Assets<Mesh>>,
                      line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
                      vs: &Vec<Vec3>,
                      color: Color,
                      name: String| {
        commands
            .spawn((
                Mesh3d(meshes.add(
                    Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(vs.iter().map(|v| v.to_array()).collect()),
                    ),
                )),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color,
                    ..Default::default()
                })),
            ))
            .insert(Name::new(name));
    };
    add_arrows(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &tangents,
        RED.into(),
        "t".to_string(),
    );
    add_arrows(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &normals,
        YELLOW.into(),
        "n".to_string(),
    );
    add_arrows(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &binormals,
        GREEN.into(),
        "b".to_string(),
    );

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
