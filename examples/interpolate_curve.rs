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

#[derive(Component)]
struct CurveContainer(NurbsCurve3D<f64>);

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, (close_on_esc, find_closest_point));
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
         _points_materials: &mut ResMut<'_, Assets<PointsMaterial>>| {
            let (min, max) = curve.knots_domain();
            let interval = max - min;
            let step = interval / 32.;
            let mut t = min;
            let mut vertices: Vec<Vec3> = vec![];
            let mut tangents: Vec<Vec3> = vec![];
            let len = 0.15;

            while t <= max {
                let pt = curve.point_at(t).cast::<f32>();
                vertices.push(pt.into());

                let tangent = curve.tangent_at(t).cast::<f32>().normalize();
                tangents.push(tangent.into());

                t += step;
            }

            let mut line = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineList, default());
            let line_list_vertices = vertices
                .iter()
                .enumerate()
                .flat_map(|(i, pt)| {
                    let tangent: Vec3 = tangents[i];
                    let normal = Vec3::new(-tangent.y, tangent.x, tangent.z);
                    [*pt, *pt + normal * len]
                })
                .map(|p| p.to_array())
                .collect();
            line.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(line_list_vertices),
            );
            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(line),
                    material: line_materials.add(LineMaterial {
                        color: Color::AQUAMARINE,
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("normal"));

            let samples = curve.tessellate(Some(1e-8));
            let mut line = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineStrip, default());
            let line_vertices = samples
                .iter()
                .map(|p| p.cast::<f32>())
                .map(|p| [p.x, p.y, p.z])
                .collect();
            line.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(line_vertices),
            );
            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(line),
                    material: line_materials.add(LineMaterial {
                        color: Color::TOMATO,
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("curve"));

            vertices
        };

    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 2.0, 0.),
        Point3::new(1.0, 2.5, 0.),
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
            ..Default::default()
        })
        .insert(Name::new("interpolation targets"));

    let interpolated = NurbsCurve::try_interpolate(&interpolation_target, 3, None, None).unwrap();

    let vertices = add_curve(
        &interpolated,
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &mut points_materials,
    );

    commands.spawn((CurveContainer(interpolated),));

    let center = if vertices.is_empty() {
        Vec3::ZERO
    } else {
        vertices.iter().fold(Vec3::ZERO, |a, b| a + *b) / (vertices.len() as f32)
    };

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
        transform: Transform::from_translation(center + Vec3::new(0., 0., 3.))
            .looking_at(center, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
}

fn find_closest_point(
    curves: Query<&CurveContainer>,
    window: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform)>,
    mut gizmos: Gizmos,
) {
    let w = window.single();
    let (camera, camera_transform) = camera.single();
    if let Some(ray) = w
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
    {
        if let Some(d) = ray.intersect_plane(Vec3::zeroed(), Plane3d::new(Vec3::Z)) {
            let pt = ray.get_point(d);
            gizmos.circle(pt, Direction3d::Z, 0.1, Color::GRAY);
            let curve = curves.single();
            let p = Point3::from(pt).cast();
            let closest = curve.0.closest_point(&p);
            let center = closest.cast::<f32>().into();
            gizmos.circle(center, Direction3d::Z, 0.05, Color::WHITE);

            gizmos.line(pt, center, Color::WHITE);
        }
    }
}
