use bevy::{
    color::palettes::css::{BLUE, GREEN, ORANGE, PURPLE, RED, WHITE},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_inspector_egui::quick::WorldInspectorPlugin;
use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use boolean::CurveVariant;
use itertools::Itertools;
use materials::*;
use nalgebra::{Rotation2, Translation2};

mod boolean;
mod materials;
pub mod misc;
mod systems;

use curvo::prelude::*;
use operation::BooleanOperation;
use systems::screenshot_on_spacebar;

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
        .add_plugins(WorldInspectorPlugin::new())
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

const OPTION: CurveIntersectionSolverOptions<f64> = CurveIntersectionSolverOptions {
    minimum_distance: 1e-4,
    knot_domain_division: 500,
    max_iters: 1000,
    step_size_tolerance: 1e-8,
    cost_tolerance: 1e-10,
};

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    commands.spawn((
        Projection::Perspective(PerspectiveProjection {
            near: 1e-3,
            ..Default::default()
        }),
        Transform::from_translation(Vec3::new(0., 0., 5.)),
        PanOrbitCamera::default(),
    ));

    // let (subject, clip) = boolean::circle_rectangle_case();
    // let (subject, clip) = boolean::periodic_interpolation_case();
    // let (subject, clip) = boolean::island_case();
    // let (subject, clip) = boolean::compound_circle_x_rectangle_case();
    // let (subject, clip) = boolean::rounded_rectangle_x_rectangle_case();
    let (subject, clip) = boolean::rectangular_annulus_x_rectangle_case();
    // let (subject, clip) = boolean::rounded_t_shape_x_rectangle_case();
    // let (subject, clip) = boolean::rounded_t_shape_x_t_shape_case();
    // let (subject, clip) = boolean::rectangular_annulus_x_rectangular_annulus_case();

    let delta: f64 = 0.0;
    let trans = Translation2::new(delta.cos(), 0.) * Rotation2::new(delta);
    let clip = clip.transformed(&trans.into());

    let spawn_curve = |commands: &mut Commands,
                       meshes: &mut ResMut<Assets<Mesh>>,
                       line_materials: &mut ResMut<Assets<LineMaterial>>,
                       transform: Transform,
                       curve: &CurveVariant,
                       color: Option<Color>,
                       sequence: bool| {
        let curves: Vec<&NurbsCurve2D<f64>> = match curve {
            CurveVariant::Curve(c) => vec![c],
            CurveVariant::Compound(c) => c.spans().iter().collect_vec(),
            CurveVariant::Region(r) => {
                let ex = r.exterior().spans().iter().collect_vec();
                r.interiors()
                    .iter()
                    .flat_map(|i| i.spans())
                    .chain(ex)
                    .collect()
            }
        };
        curves.iter().for_each(|curve| {
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
                let start = RED;
                let end = GREEN;
                line.with_inserted_attribute(
                    Mesh::ATTRIBUTE_COLOR,
                    VertexAttributeValues::Float32x4(
                        (0..n)
                            .map(|i| {
                                let t = i as f32 / (n - 1) as f32;
                                let c0 = start * (1. - t);
                                let c1 = end * t;
                                let c = c0 + c1;
                                c.to_f32_array()
                            })
                            .collect(),
                    ),
                )
            } else {
                line
            };
            commands.spawn((
                Mesh3d(meshes.add(line)),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color: color.unwrap_or(WHITE.with_alpha(0.25).into()),
                    opacity: 0.6,
                    alpha_mode: AlphaMode::Blend,
                })),
                transform,
            ));
        });
    };

    let basis = Transform::from_xyz(0., 5., 0.);

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        basis,
        &subject,
        None,
        false,
    );

    spawn_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        basis,
        &clip,
        None,
        false,
    );

    [&subject, &clip].iter().enumerate().for_each(|(i, c)| {
        let (start, end) = c.end_points();
        commands
            .spawn((
                Mesh3d(
                    meshes.add(PointsMesh {
                        vertices: [start, end]
                            .iter()
                            .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                            .collect(),
                        colors: Some(if i == 0 {
                            vec![ORANGE.into(), ORANGE.into()]
                        } else {
                            vec![PURPLE.into(), PURPLE.into()]
                        }),
                    }),
                ),
                MeshMaterial3d(points_materials.add(PointsMaterial {
                    settings: bevy_points::material::PointsShaderSettings {
                        color: WHITE.into(),
                        point_size: 0.05,
                        ..Default::default()
                    },
                    circle: true,
                    ..Default::default()
                })),
                basis,
            ))
            .insert(Name::new("End points"));
    });

    let ops = [
        BooleanOperation::Union,
        BooleanOperation::Intersection,
        BooleanOperation::Difference,
    ];
    let n = ops.len();
    let inv_n = 1. / n as f32;
    let on = inv_n * 0.5;
    let h = n as f32 * 7.5;

    ops.into_iter().enumerate().for_each(|(i, op)| {
        let fi = i as f32 * inv_n + on - 0.5;
        let tr = Transform::from_xyz(fi * h, 0., 0.);

        let clip = subject.boolean(op, &clip, Some(OPTION.clone())).unwrap();
        let regions = clip.regions();
        let info = clip.info();

        info.spans().iter().enumerate().for_each(|(i, span)| {
            spawn_curve(
                &mut commands,
                &mut meshes,
                &mut line_materials,
                tr * Transform::from_xyz(0., 0., i as f32 * 1e-1),
                &span.clone().into(),
                Some(GREEN.into()),
                true,
            );
        });

        info.node_chunks()
            .iter()
            .enumerate()
            .for_each(|(i, (a, b))| {
                let tr = Transform::from_xyz(0., 0., i as f32 * 1e-1);
                commands.spawn((
                    Mesh3d(
                        meshes.add(PointsMesh {
                            vertices: [a.0, b.0]
                                .iter()
                                .map(|p| p.cast::<f32>().coords.to_homogeneous().into())
                                .collect(),
                            colors: Some([RED.into(), BLUE.into()].to_vec()),
                        }),
                    ),
                    MeshMaterial3d(points_materials.add(PointsMaterial {
                        settings: bevy_points::material::PointsShaderSettings {
                            point_size: 0.05,
                            ..Default::default()
                        },
                        circle: true,
                        ..Default::default()
                    })),
                    tr,
                ));

                let line = Mesh::new(PrimitiveTopology::LineStrip, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(
                            [a.0, b.0]
                                .iter()
                                .map(|pt| pt.coords.cast::<f32>().to_homogeneous().into())
                                .collect(),
                        ),
                    );
                commands.spawn((
                    Mesh3d(meshes.add(line)),
                    MeshMaterial3d(line_materials.add(LineMaterial {
                        color: Color::WHITE,
                        ..Default::default()
                    })),
                    tr,
                ));
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
                            tess.vertices().iter().map(|_p| [0., 0., 1.]).collect(),
                        ),
                    )
                    .with_inserted_indices(Indices::U32(
                        tess.faces()
                            .iter()
                            .flat_map(|f| f.iter().map(|i| *i as u32))
                            .collect(),
                    ));
                commands
                    .spawn((
                        Mesh3d(meshes.add(mesh)),
                        MeshMaterial3d(normal_materials.add(NormalMaterial {
                            cull_mode: None,
                            ..Default::default()
                        })),
                        tr,
                    ))
                    .insert(Name::new("Triangulation"));
            }

            /*
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
                        &curve.clone().into(),
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
                        &curve.clone().into(),
                        Some(Color::RED),
                        false,
                    );
                });
            });
            */
        });
    });
}
