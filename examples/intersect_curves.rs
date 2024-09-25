use std::f32::consts::FRAC_PI_2;

use bevy::{
    color::palettes::{css::TOMATO, tailwind::LIME_500},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use nalgebra::{Point2, Rotation2, Translation2, Vector2};

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
        // .add_plugins(WorldInspectorPlugin::new())
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup).add_systems(Update, update);
    }
}

#[derive(Component)]
struct FirstCurve(pub NurbsCurve2D<f64>);

#[derive(Component)]
struct SecondCurve(pub NurbsCurve2D<f64>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let points = vec![
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 0.0),
        Point2::new(-1.0, 0.0),
        Point2::new(-1.0, 1.0),
        Point2::new(1.0, 1.0),
    ];
    let curve = NurbsCurve2D::try_interpolate(&points, 3).unwrap();

    let line_vertices = curve
        .tessellate(Some(1e-8))
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
                color: TOMATO.into(),
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(FirstCurve(curve.clone()))
        .insert(Name::new("curve"));

    let circle =
        NurbsCurve2D::try_circle(&Point2::origin(), &Vector2::x(), &Vector2::y(), 1.).unwrap();

    let line_vertices = circle
        .tessellate(Some(1e-8))
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
                color: LIME_500.into(),
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(SecondCurve(circle.clone()))
        .insert(Name::new("circle"));

    /*
    let traversed = BoundingBoxTraversal::try_traverse(&curve, &circle, None, None);
    if let Ok(traversed) = traversed {
        let n = traversed.pairs_iter().count();
        traversed
            .pairs_iter()
            .enumerate()
            .for_each(|(idx, (a, b))| {
                let t = (idx as f32) / (n as f32);
                let hue = t * 360. * 1e2 % 360.;
                let color = Color::hsl(hue, 0.5, 0.5);
                let b0 = a.bounding_box();
                let b1 = b.bounding_box();
                let vertices = b0
                    .lines()
                    .iter()
                    .chain(b1.lines().iter())
                    .flat_map(|(a, b)| {
                        [a, b]
                            .iter()
                            .map(|p| p.cast::<f32>())
                            .map(|p| [p.x, p.y, 0.])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let line = Mesh::new(PrimitiveTopology::LineList, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(vertices),
                    );
                commands
                    .spawn(MaterialMeshBundle {
                        mesh: meshes.add(line),
                        material: line_materials.add(LineMaterial {
                            // color: Color::WHITE,
                            color,
                            opacity: 0.5,
                            alpha_mode: AlphaMode::Blend,
                        }),
                        // visibility: Visibility::Hidden,
                        ..Default::default()
                    })
                    .insert(Name::new("bounding box"));

                [a.curve(), b.curve()].iter().for_each(|curve| {
                    let line_vertices = curve
                        .tessellate(Some(1e-8))
                        .iter()
                        .map(|p| p.cast::<f32>())
                        .map(|p| [p.x, p.y, 0.])
                        .collect();
                    let line = Mesh::new(PrimitiveTopology::LineStrip, default())
                        .with_inserted_attribute(
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
                        .insert(Name::new("segment"));
                });
            });
    }
        */

    /*
    let intersections = curve.find_intersections(&circle, None);
    if let Ok(intersections) = intersections {
        commands
            .spawn(MaterialMeshBundle {
                mesh: meshes.add(PointsMesh::from_iter(intersections.iter().flat_map(|it| {
                    [
                        Vec3::from(it.a().0.cast::<f32>().coords.to_homogeneous()),
                        Vec3::from(it.b().0.cast::<f32>().coords.to_homogeneous()),
                    ]
                }))),
                material: points_materials.add(PointsMaterial {
                    settings: PointsShaderSettings {
                        point_size: 0.025,
                        color: Color::WHITE,
                        ..Default::default()
                    },
                    circle: true,
                    ..Default::default()
                }),
                // visibility: Visibility::Hidden,
                ..Default::default()
            })
            .insert(Name::new("intersection points"));
    }
    */

    let center = Vec3::ZERO;
    let orth = Camera3dBundle {
        transform: Transform::from_translation(center + Vec3::new(0., 0., 8.))
            .looking_at(center, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
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
        Query<&Transform, With<Camera>>,
        Query<(&FirstCurve, &mut Transform), With<FirstCurve>>,
        Query<(&SecondCurve, &mut Transform), With<SecondCurve>>,
    )>,
    mut gizmos: Gizmos,
) {
    let speed = 1.0;
    let elapsed = time.elapsed_seconds() * speed;
    let delta = time.delta_seconds() * speed;

    set.p1().iter_mut().for_each(|(_, mut tr1)| {
        tr1.rotate_local_z(-delta * 0.25);
    });
    set.p2().iter_mut().for_each(|(_, mut tr2)| {
        let x = elapsed.cos() * 1.25;
        let y = elapsed.sin() * 1.25;
        *tr2 = tr2.with_translation(Vec3::new(x, y, 0.));
    });

    let c1 = set
        .p1()
        .iter()
        .map(|(c1, tr1)| {
            let trans1 = Rotation2::new(tr1.rotation.to_euler(EulerRot::XYZ).2);
            c1.0.transformed(&trans1.cast::<f64>().into())
        })
        .next()
        .unwrap();
    let c2 = set
        .p2()
        .iter()
        .map(|(c2, tr2)| {
            let trans2 = Translation2::new(tr2.translation.x, tr2.translation.y);
            c2.0.transformed(&trans2.cast::<f64>().into())
        })
        .next()
        .unwrap();

    /*
    let traversed = BoundingBoxTraversal::try_traverse(&c1, &c2, None, None);
    if let Ok(traversed) = traversed {
        let n = traversed.pairs_iter().count();
        traversed
            .pairs_iter()
            .enumerate()
            .for_each(|(idx, (a, b))| {
                let t = (idx as f32) / (n as f32);
                let hue = t * 360. * 1e2 % 360.;
                let b0 = a.bounding_box();
                let b1 = b.bounding_box();
                let color = Color::hsla(hue, 0.5, 0.5, 0.45);
                /*
                gizmos.cuboid(
                    Transform::from_translation(b0.center().to_homogeneous().cast::<f32>().into())
                        .with_scale(b0.size().to_homogeneous().cast::<f32>().into()),
                    color,
                );
                */
                gizmos.cuboid(
                    Transform::from_translation(b1.center().to_homogeneous().cast::<f32>().into())
                        .with_scale(b1.size().to_homogeneous().cast::<f32>().into()),
                    // color,
                    Color::GRAY.with_a(0.75),
                );
            });
    }
    */

    let p0 = set.p0();
    let camera_transform = p0.single();

    let intersections = c1.find_intersections(&c2, None);
    if let Ok(intersections) = intersections {
        intersections.iter().for_each(|it| {
            let p: Vec3 = it.a().0.coords.to_homogeneous().cast::<f32>().into();
            let normal = (camera_transform.translation - p).normalize();
            let dir = Dir3::new_unchecked(normal);
            gizmos.circle(p, dir, 1e-2 * 2.5, Color::WHITE);
        });
    }
}
