use std::f64::consts::TAU;

use bevy::{
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
    window::close_on_esc,
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
        app.add_systems(Startup, setup)
            .add_systems(Update, close_on_esc);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let mut rng = rand::thread_rng();
    let n = 12;
    let min_radius = 0.25;
    let max_radius = 2.5;
    let depth = 10.;

    let points: Vec<_> = (0..n)
        .map(|i| {
            let g = rng.gen::<f64>();
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
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(mesh),
            material: line_materials.add(LineMaterial {
                color: Color::WHITE,
            }),
            visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("curve"));

    let segments = curve.decompose_bezier_segments();
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
            .spawn(MaterialMeshBundle {
                mesh: meshes.add(mesh),
                material: line_materials.add(LineMaterial {
                    color: Color::hsl(t * 360., 0.5, 0.5),
                }),
                ..Default::default()
            })
            .insert(Name::new("bezier segment"));

        let (start, end) = segment.knots_domain();
        let p0 = segment.point_at(start);
        let p1 = segment.point_at(end);

        commands
            .spawn(MaterialMeshBundle {
                mesh: meshes.add(PointsMesh {
                    vertices: [p0, p1].iter().map(|pt| (*pt).into()).collect(),
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
                ..Default::default()
            })
            .insert(Name::new("interpolation targets"));
    });

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
    commands.spawn(InfiniteGridBundle::default());
}
