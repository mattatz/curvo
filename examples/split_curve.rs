use bevy::{
    color::palettes::css::{SALMON, TOMATO},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use materials::*;
use nalgebra::{Point2, Vector2};

use curvo::prelude::*;
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
            .add_systems(Update, split_animation);
    }
}

#[derive(Component)]
struct ProfileCurve(pub NurbsCurve2D<f64>);

#[derive(Component)]
struct FirstCurve;

#[derive(Component)]
struct SecondCurve;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    let control_points: Vec<Point2<f64>> = vec![
        Point2::new(-1., -1.),
        Point2::new(1., -1.),
        Point2::new(1., 0.),
        Point2::new(-1., 0.),
        Point2::new(-1., 1.),
        Point2::new(1., 1.),
    ];
    let degree = 3;
    let curve = NurbsCurve2D::try_interpolate(&control_points, degree).unwrap();
    commands.spawn(ProfileCurve(curve));

    commands.spawn((
        FirstCurve,
        Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::LineStrip, default()))),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: Color::WHITE,
            ..Default::default()
        })),
    ));
    commands.spawn((
        SecondCurve,
        Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::LineStrip, default()))),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: TOMATO.into(),
            ..Default::default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        PanOrbitCamera::default(),
    ));
    // commands.spawn(InfiniteGridBundle::default());
}

fn split_animation(
    time: Res<Time>,
    profile: Query<&ProfileCurve>,
    mut meshes: ResMut<Assets<Mesh>>,
    first: Query<&Mesh3d, With<FirstCurve>>,
    second: Query<&Mesh3d, With<SecondCurve>>,
    mut gizmos: Gizmos,
) {
    let profile = profile.single();
    let (start, end) = profile.0.knots_domain();
    let sec = time.elapsed_secs_f64();
    let t = start + (end - start) * (0.5 + 0.5 * sec.sin());
    let (c0, c1) = profile.0.try_split(t).unwrap();
    let first = first.single();
    let second = second.single();

    let tesselate = |curve: &NurbsCurve2D<f64>| {
        curve
            .tessellate(Some(1e-6))
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, 0.])
            .collect::<Vec<_>>()
    };

    meshes.get_mut(first).unwrap().insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(tesselate(&c0)),
    );
    meshes.get_mut(second).unwrap().insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(tesselate(&c1)),
    );

    let (start, end) = c1.knots_domain();
    let pts = c1.sample_regular_range_with_parameter(start, end, 32);
    pts.iter().for_each(|(u, pt)| {
        let tangent = c1.tangent_at(*u).normalize();
        let p0 = pt.cast::<f32>();
        let t = tangent.cast::<f32>();
        let n = Vector2::new(-t.y, t.x);
        let p1 = p0 + n * 0.15;
        gizmos.linestrip(
            [Vec3::new(p0.x, p0.y, 0.), Vec3::new(p1.x, p1.y, 0.)],
            SALMON,
        );
    });
}
