use bevy::{
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    window::close_on_esc,
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{Rotation2, Translation2};

mod boolean;
mod materials;
mod systems;

use curvo::prelude::*;
use operation::BooleanOperation;
use status::Status;
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
    let camera = Camera3dBundle {
        projection: Projection::Perspective(PerspectiveProjection {
            near: 1e-3,
            ..Default::default()
        }),
        transform: Transform::from_translation(Vec3::new(0., 0., 5.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));

    let (subject, clip) = boolean::circle_rectangle_case();
    let (subject, clip) = boolean::periodic_interpolation_case();
    let (subject, clip) = boolean::island_case();

    /*
    let delta: f64 = 0.46303153026000005;
    let delta: f64 = 0.46544134026; // error with degeneracies
    let delta: f64 = 4.38834548724;
    let delta: f64 = 1.27497772974;
    */

    // let delta: f64 = 0.0;
    let delta: f64 = 0.75;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    // let clip = clip.transformed(&trans.into());

    let spawn_curve = |commands: &mut Commands,
                       meshes: &mut ResMut<Assets<Mesh>>,
                       line_materials: &mut ResMut<Assets<LineMaterial>>,
                       transform: Transform,
                       curve: &NurbsCurve2D<f64>,
                       color: Option<Color>,
                       sequence: bool| {
        let samples = curve.tessellate(Some(1e-8));
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, 0.])
            .collect_vec();
        let n = line_vertices.len();
        let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
        );
        let line = if sequence {
            let start = Color::RED;
            let end = Color::GREEN;
            line.with_inserted_attribute(
                Mesh::ATTRIBUTE_COLOR,
                VertexAttributeValues::Float32x4(
                    (0..n)
                        .map(|i| {
                            let t = i as f32 / (n - 1) as f32;
                            let c0 = start * (1. - t);
                            let c1 = end * t;
                            (c0 + c1).as_rgba_f32()
                        })
                        .collect(),
                ),
            )
        } else {
            line
        };
        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(line),
            material: line_materials.add(LineMaterial {
                color: color.unwrap_or(Color::WHITE.with_a(0.25)),
                opacity: 0.6,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            }),
            transform,
            ..Default::default()
        });
    };

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        Transform::default(),
        &subject,
        None,
        false,
    );
    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        Transform::default(),
        &clip,
        None,
        false,
    );

    let option = CurveIntersectionSolverOptions {
        // minimum_distance: 1e-5,
        // knot_domain_division: 1000,
        // max_iters: 2000,
        ..Default::default()
    };

    /*
    let traversed = BoundingBoxTraversal::try_traverse(
        &subject,
        &clip,
        Some(subject.knots_domain_interval() / (option.knot_domain_division as f64)),
        Some(clip.knots_domain_interval() / (option.knot_domain_division as f64)),
    );
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
                [&b0, &b1].iter().enumerate().for_each(|(j, bb)| {
                    let color = if j == 0 { Color::RED } else { Color::BLUE };
                    let vertices = bb
                        .lines()
                        .iter()
                        // .chain(b1.lines().iter())
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
                });

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
                            visibility: Visibility::Hidden,
                            ..Default::default()
                        })
                        .insert(Name::new("segment"));
                });
            });
    }
    */

    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(PointsMesh {
            vertices: [
                subject.point_at(subject.knots_domain().0),
                clip.point_at(clip.knots_domain().0),
            ]
            .iter()
            .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
            .collect(),
            colors: Some(vec![Color::RED, Color::BLUE]),
            ..Default::default()
        }),
        material: points_materials.add(PointsMaterial {
            settings: bevy_points::material::PointsShaderSettings {
                color: Color::WHITE,
                point_size: 0.025,
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        }),
        ..Default::default()
    });

    let ops = [
        // BooleanOperation::Union,
        // BooleanOperation::Intersection,
        BooleanOperation::Difference,
    ];
    let n = ops.len();
    let inv_n = 1. / n as f32;
    let on = inv_n * 0.5;
    let h = n as f32 * 2.5;

    let intersections = subject
        .find_intersections(&clip, Some(option.clone()))
        .unwrap();
    println!("intersections: {:?}", intersections.len());

    let points = intersections
        .iter()
        .enumerate()
        .flat_map(|(i, it)| {
            let pa = it.a().0.cast::<f32>();
            let pb = it.b().0.cast::<f32>();
            vec![Vec3::new(pa.x, pa.y, 0.), Vec3::new(pb.x, pb.y, 0.)]
        })
        .collect();

    commands.spawn(MaterialMeshBundle {
        mesh: meshes.add(PointsMesh {
            vertices: points,
            ..Default::default()
        }),
        material: points_materials.add(PointsMaterial {
            settings: bevy_points::material::PointsShaderSettings {
                color: Color::WHITE,
                point_size: 0.0025,
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        }),
        ..Default::default()
    });

    /*
    let parameter_eps = 1e-3;
    intersections.iter().for_each(|it| {
        let a0 = subject.point_at(it.a().1 - parameter_eps);
        let a1 = subject.point_at(it.a().1 + parameter_eps);
        let b0 = clip.point_at(it.b().1 - parameter_eps);
        let b1 = clip.point_at(it.b().1 + parameter_eps);
        let intersected = Line::new(a0, a1).intersects(&Line::new(b0, b1));
        [[a0, a1], [b0, b1]]
            .iter()
            .enumerate()
            .for_each(|(i, line)| {
                let line = Mesh::new(PrimitiveTopology::LineStrip, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(
                            line.iter()
                                .map(|pt| pt.coords.cast::<f32>().to_homogeneous().into())
                                .collect(),
                        ),
                    );
                commands.spawn(MaterialMeshBundle {
                    mesh: meshes.add(line),
                    material: line_materials.add(LineMaterial {
                        color: if intersected {
                            if i == 0 {
                                Color::RED
                            } else {
                                Color::YELLOW
                            }
                        } else if i == 0 {
                            Color::RED
                        } else {
                            Color::BLUE
                        },
                        ..Default::default()
                    }),
                    ..Default::default()
                });
            });
    });
    */

    ops.into_iter().enumerate().for_each(|(i, op)| {
        let (regions, intersections) = subject.boolean(op, &clip, Some(option.clone())).unwrap();
        let fi = i as f32 * inv_n + on - 0.5;
        let tr = Transform::from_xyz(fi * h, 0., 0.);

        let points = intersections
            .iter()
            .enumerate()
            .map(|(i, it)| {
                let pt = it.vertex().position().cast::<f32>();
                // tr * Vec3::new(pt.x, pt.y, i as f32 * 1e-1)
                tr * Vec3::new(pt.x, pt.y, 0.)
            })
            .collect();

        let colors = intersections
            .iter()
            .map(|it| match it.status() {
                Status::None => Color::WHITE,
                Status::Enter => Color::BLUE,
                Status::Exit => Color::RED,
            })
            .collect();

        commands.spawn(MaterialMeshBundle {
            mesh: meshes.add(PointsMesh {
                vertices: points,
                colors: Some(colors),
            }),
            material: points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: Color::WHITE,
                    // point_size: 0.025,
                    point_size: 0.05,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            }),
            visibility: Visibility::Hidden,
            ..Default::default()
        });

        regions.iter().for_each(|region| {
            let tess = region.tessellate(None);
            if let Ok(tess) = tess {
                let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(
                            tess.vertices()
                                .iter()
                                .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                                .collect(),
                        ),
                    )
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_NORMAL,
                        VertexAttributeValues::Float32x3(
                            tess.vertices().iter().map(|p| [0., 0., 1.]).collect(),
                        ),
                    )
                    .with_inserted_indices(Indices::U32(
                        tess.faces()
                            .iter()
                            .flat_map(|f| f.iter().map(|i| *i as u32))
                            .collect(),
                    ));

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
                    .insert(Name::new("lofted"));
            }

            region
                .exterior()
                .spans()
                .iter()
                .enumerate()
                .for_each(|(i, curve)| {
                    spawn_curve(
                        &mut commands,
                        &mut meshes,
                        &mut line_materials,
                        tr * Transform::from_xyz(0., 0., i as f32 * 1e-1),
                        // tr,
                        curve,
                        Some(Color::GREEN),
                        true,
                    );
                });
            region.interiors().iter().for_each(|interior| {
                interior.spans().iter().for_each(|curve| {
                    spawn_curve(
                        &mut commands,
                        &mut meshes,
                        &mut line_materials,
                        tr,
                        curve,
                        Some(Color::RED),
                        false,
                    );
                });
            });
        });
    });
}
