use bevy::{
    color::palettes::css::{BLUE, LIME, TOMATO, WHITE},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial, prelude::PointsMesh};
use nalgebra::Point2;

use curvo::prelude::*;

mod materials;

use materials::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (800., 600.).into(),
                title: "Catmull-Rom Spline".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(LineMaterialPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
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
    // Define control points for the Catmull-Rom spline
    let points = vec![
        Point2::new(-2.0, -1.0),
        Point2::new(-1.0, 1.5),
        Point2::new(1.0, -1.5),
        Point2::new(2.0, 1.0),
        Point2::new(0.0, 2.0),
    ];

    // Visualize the input points (white, larger)
    commands
        .spawn((
            Mesh3d(
                meshes.add(PointsMesh {
                    vertices: points
                        .iter()
                        .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                        .collect(),
                    ..Default::default()
                }),
            ),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: WHITE.into(),
                    point_size: 0.08,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            })),
        ))
        .insert(Name::new("input points"));

    // Create Catmull-Rom curves with different tension values
    let tensions = [
        (0.25, TOMATO, "tension=0.25 (loose)"),
        (0.5, LIME, "tension=0.5 (standard)"),
        (1.0, BLUE, "tension=1.0 (tight)"),
    ];

    for (tension, color, name) in tensions {
        let curve = NurbsCurve2D::catmull_rom(&points, tension);

        // Tessellate and render the curve
        let samples = curve.tessellate(Some(1e-8));
        let line_vertices: Vec<_> = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, 0.])
            .collect();

        let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
        );

        commands
            .spawn((
                Mesh3d(meshes.add(line)),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color: color.into(),
                    ..Default::default()
                })),
            ))
            .insert(Name::new(name));

        // Visualize control points for this curve (smaller, same color)
        let control_pts: Vec<_> = curve
            .dehomogenized_control_points()
            .iter()
            .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
            .collect();

        commands
            .spawn((
                Mesh3d(meshes.add(PointsMesh {
                    vertices: control_pts,
                    ..Default::default()
                })),
                MeshMaterial3d(points_materials.add(PointsMaterial {
                    settings: bevy_points::material::PointsShaderSettings {
                        color: color.into(),
                        point_size: 0.03,
                        ..Default::default()
                    },
                    circle: true,
                    ..Default::default()
                })),
            ))
            .insert(Name::new(format!("{} control points", name)));
    }

    // Setup camera
    let scale = 5.;
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 1.,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(Vec3::new(0., 0., 3.)).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
