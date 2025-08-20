use std::f32::consts::FRAC_PI_2;

use bevy::{
    color::palettes::css::{CORNFLOWER_BLUE, TOMATO},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use nalgebra::{Point3, Rotation3, Vector3};

use curvo::prelude::*;

mod materials;

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
        app.add_systems(Startup, setup).add_systems(Update, update);
    }
}

#[derive(Component)]
struct Curve(pub NurbsCurve3D<f64>);

#[derive(Component)]
struct PlaneCollider;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut mesh_materials: ResMut<Assets<StandardMaterial>>,
) {
    let points = vec![
        Point3::new(-1.0, -1.0, -1.0),
        Point3::new(1.0, -1.0, 1.0),
        Point3::new(1.0, 0.0, -1.0),
        Point3::new(-1.0, 0.0, 1.0),
        Point3::new(-1.0, 1.0, -1.0),
        Point3::new(1.0, 1.0, 1.0),
    ];
    let curve = NurbsCurve3D::interpolate(&points, 3).unwrap();

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
        .spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: TOMATO.into(),
                ..Default::default()
            })),
        ))
        .insert(Curve(curve.clone()))
        .insert(Name::new("curve"));

    // add plane mesh
    let pl = Plane3d::new(Dir3::Z.into(), Vec2::splat(2.));
    commands.spawn((
        Mesh3d(meshes.add(pl)),
        MeshMaterial3d(mesh_materials.add(StandardMaterial {
            base_color: CORNFLOWER_BLUE.with_alpha(0.1).into(),
            unlit: true,
            cull_mode: None,
            double_sided: true,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        })),
        Transform::from_translation(Vec3::new(0., 0., 0.)),
        PlaneCollider,
        Name::new("plane"),
    ));

    let center = Vec3::ZERO;
    commands.spawn((
        Transform::from_translation(center + Vec3::new(0., 0., 8.)).looking_at(center, Vec3::Y),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            x_axis_color: Color::BLACK,
            z_axis_color: Color::BLACK,
            ..Default::default()
        },
        transform: Transform::from_rotation(Quat::from_rotation_x(FRAC_PI_2)),
        ..Default::default()
    });
}

#[allow(clippy::type_complexity)]
fn update(
    time: Res<Time>,
    mut set: ParamSet<(
        Query<(&Curve, &mut Transform), With<Curve>>,
        Query<&mut Transform, With<PlaneCollider>>,
        Query<&Transform, With<Camera>>,
    )>,
    mut gizmos: Gizmos,
) {
    let speed = 1.0;
    let elapsed = time.elapsed_secs() * speed;
    let delta = time.delta_secs() * speed;

    set.p0().iter_mut().for_each(|(_, mut tr1)| {
        tr1.rotate_local_z(-delta * 0.25);
        tr1.rotate_local_x(delta * 0.1);
    });

    let curve = set
        .p0()
        .iter()
        .map(|(c1, tr1)| {
            let (axis, angle) = tr1.rotation.to_axis_angle();
            let axis = axis * angle;
            let trans1 = Rotation3::from_scaled_axis(axis.into());
            c1.0.transformed(&trans1.cast::<f64>().into())
        })
        .next()
        .unwrap();

    let normal = Vector3::z();
    let constant = elapsed.cos() as f64 * 1.25;
    let plane = Plane::new(normal, constant);

    {
        let mut p1 = set.p1();
        let mut plane_tr = p1.single_mut().unwrap();
        plane_tr.translation = Vec3::new(0., 0., -constant as f32);
    }

    let its = curve.find_intersection(&plane, None);
    let p2 = set.p2();
    let camera_transform = p2.single().unwrap();
    if let Ok(its) = its {
        its.iter().for_each(|it| {
            let p: Vec3 = it.a().0.coords.cast::<f32>().into();
            let tr = Isometry3d::new(p, camera_transform.rotation);
            gizmos.circle(tr, 1e-2 * 5.0, Color::WHITE);
        });
    }
}
