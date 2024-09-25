use std::f64::consts::TAU;

use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use nalgebra::{Point3, Vector3};

use curvo::prelude::*;
use rand::Rng;
use systems::screenshot_on_spacebar;
mod materials;
mod systems;

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
            .add_systems(Update, screenshot_on_spacebar);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let mut rng = rand::thread_rng();
    let n = 10;
    let depth = 8.;
    let min_radius = 1.0;
    let max_radius = 2.5;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.gen::<f64>();
            let t = i as f64 / n as f64;
            let rad = min_radius + g * (max_radius - min_radius);
            let r = 0_f64;
            let x = r.cos() * rad;
            let z = r.sin() * rad;
            let y = depth * t;
            Point3::new(x, y, z)
        })
        .collect();
    let profile = NurbsCurve3D::try_interpolate(&points, 3).unwrap();

    let profile_vertices = profile
        .tessellate(Some(1e-3))
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, p.z])
        .collect();
    let profile_mesh = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(profile_vertices),
    );
    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(profile_mesh),
            material: line_materials.add(LineMaterial {
                color: Color::WHITE,
                ..Default::default()
            }),
            ..Default::default()
        })
        .insert(Name::new("profile"));

    let revolved =
        NurbsSurface::try_revolve(&profile, &Point3::origin(), &Vector3::y(), TAU / 4.0 * 3.0)
            .unwrap();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    let option = AdaptiveTessellationOptions {
        norm_tolerance: 1e-1 * 0.5,
        ..Default::default()
    };
    let tess = revolved.tessellate(Some(option));
    let tess = tess.cast::<f32>();
    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_NORMAL,
        VertexAttributeValues::Float32x3(normals),
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs));
    mesh.insert_indices(Indices::U32(indices));

    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(mesh),
            material: normal_materials.add(NormalMaterial {
                cull_mode: None,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("swept"));

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(18., 18., 18.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
    commands.spawn(InfiniteGridBundle::default());
}
