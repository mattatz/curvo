use bevy::{
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
    window::close_on_esc,
};
use bevy_infinite_grid::InfiniteGridPlugin;
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use systems::screenshot_on_spacebar;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use nalgebra::{Point2, Vector2};

use curvo::prelude::*;

mod materials;
mod systems;

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
        app.add_systems(Startup, setup)
            .add_systems(Update, (screenshot_on_spacebar, close_on_esc));
    }
}

#[derive(Debug)]
enum CurveVariant {
    EllipseArc,
    Ellipse,
    Arc,
    Circle,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    let radius = 1.;
    let pad = radius;
    let width = radius * 2. + pad;
    let offset = -3. * width * 0.5;
    let x_axis = Vector2::x();
    let y_axis = Vector2::y();

    [
        CurveVariant::EllipseArc,
        CurveVariant::Ellipse,
        CurveVariant::Arc,
        CurveVariant::Circle,
    ]
    .iter()
    .enumerate()
    .for_each(|(i, variant)| {
        let x = offset + width * (i as f64);
        let center = Point2::new(x, 0.);
        let curve = match variant {
            CurveVariant::EllipseArc => NurbsCurve2D::try_ellipse_arc(
                &center,
                &(x_axis * 0.75),
                &(y_axis * 2.2),
                FRAC_PI_2,
                TAU,
            )
            .unwrap(),
            CurveVariant::Ellipse => {
                NurbsCurve2D::try_ellipse(&center, &(x_axis * 0.75), &(y_axis * 2.2)).unwrap()
            }
            CurveVariant::Arc => {
                NurbsCurve2D::try_arc(&center, &x_axis, &y_axis, radius, -FRAC_PI_2, PI).unwrap()
            }
            CurveVariant::Circle => {
                NurbsCurve2D::try_circle(&center, &x_axis, &y_axis, radius).unwrap()
            }
        };

        let samples = curve.tessellate(Some(1e-8));
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, 0.])
            .collect();
        let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
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
            .insert(Name::new(format!("{:?}", variant)));
    });

    let scale = 6.5;
    let orth = Camera3dBundle {
        projection: OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical(2.0),
            ..Default::default()
        }
        .into(),
        transform: Transform::from_translation(Vec3::new(0., 0., 4.2))
            .looking_at(Vec3::ZERO, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
}
