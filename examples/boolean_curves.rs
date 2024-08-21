use std::{f64::consts::FRAC_PI_2, thread::spawn};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        view::screenshot::ScreenshotManager,
    },
    window::{close_on_esc, PrimaryWindow},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use materials::*;
use nalgebra::{Point2, Point3, Rotation3, Transform3, Translation3, Vector2, Vector3};

mod materials;
mod systems;

use curvo::prelude::*;
use systems::screenshot_on_spacebar;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (864., 480.).into(),
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
            .add_systems(Update, (screenshot_on_spacebar, close_on_esc));
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let circle =
        NurbsCurve2D::<f64>::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.)
            .unwrap();

    let rectangle = NurbsCurve2D::<f64>::polyline(&vec![
        Point2::new(0., 0.5),
        Point2::new(2., 0.5),
        Point2::new(2., -0.5),
        Point2::new(0., -0.5),
        Point2::new(0., 0.5),
    ]);

    let spawn_curve = |commands: &mut Commands,
                       meshes: &mut ResMut<Assets<Mesh>>,
                       line_materials: &mut ResMut<Assets<LineMaterial>>,
                       curve: &NurbsCurve2D<f64>,
                       depth: f32,
                       color: Option<Color>| {
        let samples = curve.tessellate(Some(1e-8));
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, depth])
            .collect();
        let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
        );
        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: color.unwrap_or(Color::WHITE.with_a(0.25)),
                opacity: 0.6,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            }),
            ..Default::default()
        });
    };
    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &circle,
        0.,
        None,
    );
    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &rectangle,
        0.,
        None,
    );

    let mut intersections = circle
        .find_intersections(&rectangle, Default::default())
        .unwrap();
    assert!(!intersections.is_empty());
    assert!(intersections.len() % 2 == 0);

    intersections.sort_by(|i0, i1| i0.a().1.partial_cmp(&i1.a().1).unwrap());

    let points = intersections
        .iter()
        .map(|it| {
            let pt = it.a().0.cast::<f32>();
            [pt.x, pt.y, 0.].into()
        })
        .collect();

    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(PointsMesh {
            vertices: points,
            colors: Some(
                intersections
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let hue = i as f32 / intersections.len() as f32;
                        Color::hsl(hue * 360., 0.5, 0.5)
                    })
                    .collect(),
            ),
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
    });

    let mut trimmed_circle = circle.clone();
    let (s, _) = trimmed_circle.knots_domain();
    let start = trimmed_circle.point_at(s);

    let n = intersections.len();
    intersections.iter().enumerate().for_each(|(i, it)| {
        let (head, tail) = trimmed_circle.try_trim(it.a().1).unwrap();
        if i == n - 1 {
            trimmed_circle = head;
        } else {
            trimmed_circle = tail;
        }
    });

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &trimmed_circle,
        0.1,
        Some(Color::RED),
    );

    let mut trimmed_rectangle = rectangle.clone();
    let start = trimmed_rectangle.point_at(trimmed_rectangle.knots_domain().0);
    // let contains = circle.contains(&start, None).unwrap();

    let n = intersections.len();
    intersections.iter().enumerate().for_each(|(i, it)| {
        let (head, tail) = trimmed_rectangle.try_trim(it.b().1).unwrap();
        if i == n - 1 {
            trimmed_rectangle = head;
        } else {
            trimmed_rectangle = tail;
        }
    });

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &trimmed_rectangle,
        0.1,
        Some(Color::RED),
    );

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 0., 5.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
}
