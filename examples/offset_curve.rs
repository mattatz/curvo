use bevy::{
    color::palettes::css::{TOMATO, WHITE, YELLOW},
    prelude::*,
    render::camera::ScalingMode,
};
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::{PointsMaterial, PointsShaderSettings},
    mesh::PointsMesh,
    plugin::PointsPlugin,
};
use itertools::Itertools;
use misc::add_curve;
use nalgebra::Point2;

use curvo::prelude::*;

mod materials;
mod misc;

use materials::*;
use misc::*;

#[derive(Resource)]
struct Setting {
    pub corner_type: CurveOffsetCornerType,
    pub distance: f64,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            corner_type: CurveOffsetCornerType::Round,
            distance: -0.1,
        }
    }
}

#[derive(Component)]
struct ProfileCurve(pub NurbsCurve2D<f64>);

#[derive(Component)]
struct OffsetVertex(pub CompoundCurve2D<f64>);

#[derive(Component)]
struct OffsetCurve(pub CompoundCurve2D<f64>);

impl OffsetCurve {
    pub fn update(&mut self, entity: CompoundCurve2D<f64>) {
        self.0 = entity;
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
        .add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: true,
        })
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(Setting::default())
            .add_systems(Startup, setup)
            .add_systems(
                PreUpdate,
                (absorb_egui_inputs,)
                    .after(bevy_egui::input::write_egui_input_system)
                    .before(bevy_egui::begin_pass_system),
            )
            .add_systems(Update, update_curve.run_if(resource_changed::<Setting>))
            .add_systems(EguiContextPass, update_ui)
            .add_systems(Update, gizmos_offset_curve);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    settings: Res<Setting>,
) {
    let points = vec![
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 1.0),
        Point2::new(-1.0, 1.0),
        Point2::new(-1.0, 3.0),
        Point2::new(1.0, 3.0),
    ];

    commands
        .spawn((
            Mesh3d(
                meshes.add(PointsMesh {
                    vertices: points
                        .iter()
                        .map(|pt| pt.cast::<f32>())
                        .map(|pt| Vec3::new(pt.x, pt.y, 0.))
                        .collect(),
                    ..Default::default()
                }),
            ),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: PointsShaderSettings {
                    point_size: 0.05,
                    color: TOMATO.into(),
                    ..Default::default()
                },
                ..Default::default()
            })),
        ))
        .insert(Name::new("points"));

    let curve = NurbsCurve2D::polyline(&points, true);
    // let curve = NurbsCurve2D::try_interpolate(&points, 2).unwrap();
    add_curve(
        &curve.elevate_dimension(),
        Some(WHITE.into()),
        Some(1e-8),
        &mut commands,
        &mut meshes,
        &mut line_materials,
    );

    let option = CurveOffsetOption::default()
        .with_corner_type(CurveOffsetCornerType::Round)
        .with_distance(settings.distance)
        .with_normal_tolerance(1e-4);
    let offset_curve = curve.offset(option.clone()).unwrap();
    let offset_curve_vertex = curve
        .offset(option.clone().with_corner_type(CurveOffsetCornerType::None))
        .unwrap();

    commands.spawn((ProfileCurve(curve),));
    commands.spawn((OffsetCurve(offset_curve),));

    commands.spawn((OffsetVertex(offset_curve_vertex),));

    let scale = 5.;
    commands.spawn((
        Projection::Orthographic(OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 2.,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(Vec3::new(0., 0., 3.)).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn update_curve(
    settings: Res<Setting>,
    profile: Query<&ProfileCurve>,
    mut offset_curve: Query<&mut OffsetCurve>,
    mut offset_vertex: Query<&mut OffsetVertex>,
) {
    let profile = profile.single().unwrap();
    let mut offset_curve = offset_curve.single_mut().unwrap();
    let option = CurveOffsetOption::default()
        .with_corner_type(settings.corner_type.clone())
        .with_distance(settings.distance)
        .with_normal_tolerance(1e-4);
    let res = profile.0.offset(option);
    if let Ok(res) = res {
        offset_curve.update(res);
    }

    let option = CurveOffsetOption::default()
        .with_corner_type(CurveOffsetCornerType::None)
        .with_distance(settings.distance)
        .with_normal_tolerance(1e-4);
    let res = profile.0.offset(option);
    if let Ok(res) = res {
        offset_vertex.single_mut().unwrap().0 = res;
    }
}

fn update_ui(mut contexts: EguiContexts, mut settings: ResMut<Setting>) {
    egui::Window::new("offset curve")
        .collapsible(false)
        .drag_to_scroll(false)
        .default_width(420.)
        .min_width(420.)
        .max_width(420.)
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("corner type");
            ui.group(|g| {
                g.horizontal(|ui| {
                    ui.selectable_value(
                        &mut settings.corner_type,
                        CurveOffsetCornerType::Sharp,
                        "Sharp",
                    );
                    ui.selectable_value(
                        &mut settings.corner_type,
                        CurveOffsetCornerType::Round,
                        "Round",
                    );
                    ui.selectable_value(
                        &mut settings.corner_type,
                        CurveOffsetCornerType::Smooth,
                        "Smooth",
                    );
                    ui.selectable_value(
                        &mut settings.corner_type,
                        CurveOffsetCornerType::Chamfer,
                        "Chamfer",
                    );
                });
            });

            ui.heading("distance");
            ui.group(|g| {
                g.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut settings.distance).speed(1e-2));
                });
            });
        });
}

fn gizmos_offset_curve(
    offset_curve: Query<&OffsetCurve>,
    offset_vertex: Query<&OffsetVertex>,
    mut gizmos: Gizmos,
) {
    let offset_curve = offset_curve.single().unwrap();
    offset_curve.0.spans().iter().for_each(|span| {
        let c = span.elevate_dimension();
        let tess = c
            .tessellate(Some(1e-8))
            .into_iter()
            .map(|p| p.cast::<f32>().into())
            .collect_vec();
        gizmos.linestrip(tess, WHITE);
    });

    /*
    let offset_vertex = offset_vertex.single().unwrap();
    offset_vertex.0.spans().iter().for_each(|span| {
        let c = span.elevate_dimension();
        let tess = c
            .tessellate(None)
            .into_iter()
            .map(|p| p.cast::<f32>().into())
            .collect_vec();
        gizmos.linestrip(tess, YELLOW);
    });
    */
}
