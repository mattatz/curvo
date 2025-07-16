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
use itertools::Itertools;
use materials::*;
use nalgebra::{Matrix4, Point3};

use curvo::prelude::*;
use rand::Rng;
mod materials;

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
    let min_radius = 1.5;
    let max_radius = 2.0;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.random::<f64>();
            let t = i as f64 / n as f64;
            let r = t * TAU;
            let rad = min_radius + g * (max_radius - min_radius);
            let x = r.cos() * rad;
            let y = r.sin() * rad;
            Point3::new(x, y, 0.)
        })
        .collect();
    let curve = NurbsCurve3D::try_periodic_interpolate(&points, 3, KnotStyle::Uniform).unwrap();

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
    let samples = 256;
    let span = (end - start) / ((samples - 1) as f64);
    let parameters: Vec<_> = (0..samples).map(|i| start + span * (i as f64)).collect();
    let curvatures = parameters
        .iter()
        .map(|parameter| {
            let p = curve.point_at(*parameter).cast::<f32>();
            let c = curve.curvature_at(*parameter).unwrap().cast::<f32>();
            (p, c)
        })
        .collect_vec();

    let mut tangents = vec![];
    let mut normals = vec![];

    let length = 0.15;
    curvatures.iter().enumerate().for_each(|(i, (p, c))| {
        let p: Vec3 = (*p).into();
        let tv: Vec3 = c.tangent_vector().into();
        let cv: Vec3 = c.curvature_vector().into();
        tangents.push(p);
        tangents.push(p + tv * length);
        normals.push(p);
        normals.push(p + cv * length);
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

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 5.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
