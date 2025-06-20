use bevy::{
    color::palettes::css::{BLUE, LIGHT_GREEN, WHITE, YELLOW},
    prelude::*,
    render::camera::ScalingMode,
};
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use itertools::Itertools;
use nalgebra::Point2;

use curvo::prelude::*;

mod boolean;
mod materials;
mod misc;

use boolean::*;
use materials::*;
use misc::*;

#[derive(Resource, Debug)]
struct Setting {
    pub corner_type: CurveOffsetCornerType,
    pub distance: f64,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            // corner_type: CurveOffsetCornerType::Sharp,
            corner_type: CurveOffsetCornerType::Round,
            // corner_type: CurveOffsetCornerType::Smooth,
            // distance: 0.2,
            distance: 0.2,
        }
    }
}

#[derive(Component)]
struct ProfileCurve(pub CompoundCurve2D<f64>);

#[derive(Component)]
struct OffsetVertex(pub Vec<CompoundCurve2D<f64>>);

#[derive(Component)]
struct OffsetCurve(pub Vec<CompoundCurve2D<f64>>);

impl OffsetCurve {
    pub fn update(&mut self, entities: Vec<CompoundCurve2D<f64>>) {
        self.0 = entities;
    }
}

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
            .add_systems(EguiContextPass, update_ui)
            .add_systems(Update, gizmos_offset_curve);
    }
}

fn setup(mut commands: Commands, settings: Res<Setting>) {
    // s-shape
    // square
    let _spans = [
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 1.0),
        Point2::new(-1.0, 1.0),
    ]
    .into_iter()
    .cycle()
    .take(5)
    .collect_vec()
    .windows(2)
    .map(|w| NurbsCurve2D::polyline(w, true))
    .collect_vec();

    // convex polygon
    let points = [
        Point2::new(-0.5, 2.),
        Point2::new(0.5, 2.),
        Point2::new(0.5, 1.),
        Point2::new(1.5, 1.),
        Point2::new(1.5, -1.),
        Point2::new(-1.5, -1.),
        Point2::new(-1.5, 1.),
        Point2::new(-0.5, 1.),
        Point2::new(-0.5, 2.),
    ];
    let spans = points
        .windows(2)
        .map(|w| NurbsCurve2D::polyline(w, true))
        .collect_vec();
    let _curve = CompoundCurve::try_new(spans).unwrap();

    let curve = match compound_rounded_t_shape() {
        boolean::CurveVariant::Compound(c) => c.inverse(),
        _ => todo!(),
    };

    let option = CurveOffsetOption::default()
        .with_corner_type(settings.corner_type)
        .with_distance(settings.distance)
        .with_normal_tolerance(1e-4);
    let offset_curve = curve.offset(option.clone()).unwrap();
    let offset_curve_vertex = curve
        .offset(option.clone().with_corner_type(CurveOffsetCornerType::None))
        .unwrap();
    // println!("{:?}", offset_curve);
    // println!("{:?}", offset_curve_vertex);

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
        PanOrbitCamera {
            orbit_sensitivity: 0.0,
            ..Default::default()
        },
    ));
}

fn update_ui(
    mut contexts: EguiContexts,
    mut settings: ResMut<Setting>,
    profile: Query<&mut ProfileCurve>,
    mut offset_curve: Query<&mut OffsetCurve>,
    mut offset_vertex: Query<&mut OffsetVertex>,
) {
    let mut changed = false;
    egui::Window::new("offset compound curve")
        .collapsible(false)
        .drag_to_scroll(false)
        .default_width(420.)
        .min_width(420.)
        .max_width(420.)
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("corner type");
            ui.group(|g| {
                g.horizontal(|ui| {
                    changed |= ui
                        .selectable_value(
                            &mut settings.corner_type,
                            CurveOffsetCornerType::Sharp,
                            "Sharp",
                        )
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut settings.corner_type,
                            CurveOffsetCornerType::Round,
                            "Round",
                        )
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut settings.corner_type,
                            CurveOffsetCornerType::Smooth,
                            "Smooth",
                        )
                        .changed();
                    changed |= ui
                        .selectable_value(
                            &mut settings.corner_type,
                            CurveOffsetCornerType::Chamfer,
                            "Chamfer",
                        )
                        .changed();
                });
            });

            ui.heading("distance");
            ui.group(|g| {
                g.horizontal(|ui| {
                    changed |= ui
                        .add(egui::DragValue::new(&mut settings.distance).speed(1e-2))
                        .changed();
                });
            });
        });

    if changed {
        let profile = profile.single().unwrap();

        let mut offset_curve = offset_curve.single_mut().unwrap();
        let option = CurveOffsetOption::default()
            .with_corner_type(settings.corner_type)
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
}

fn gizmos_offset_curve(
    profile: Query<&ProfileCurve>,
    offset_curve: Query<&OffsetCurve>,
    offset_vertex: Query<&OffsetVertex>,
    mut gizmos: Gizmos,
) {
    let tol = 1e-7 * 0.5;

    let profile = profile.single().unwrap();
    let tess = profile
        .0
        .tessellate(Some(tol))
        .into_iter()
        .map(|p| p.coords.cast::<f32>().to_homogeneous().into())
        .collect_vec();
    gizmos.linestrip(tess, WHITE);

    let offset_curve = offset_curve.single().unwrap();

    offset_curve.0.iter().for_each(|c| {
        let tess: Vec<Vec3> = c
            .tessellate(Some(tol))
            .into_iter()
            .map(|p| p.coords.cast::<f32>().to_homogeneous().into())
            .collect_vec();
        let n = tess.len();
        let c0 = Oklaba::from(YELLOW);
        let c1 = Oklaba::from(BLUE);
        let pts = tess.into_iter().enumerate().map(|(i, p)| {
            let t = (i as f32) / (n as f32);
            let c = c0.mix(&c1, t);
            (p, c)
        });
        gizmos.linestrip_gradient(pts);

        c.spans()
            .iter()
            .flat_map(|c| c.dehomogenized_control_points())
            .for_each(|p| {
                let p: Vec3 = p.coords.cast::<f32>().to_homogeneous().into();
                gizmos.sphere(p, 0.035, LIGHT_GREEN);
            });
    });

    let offset_vertex = offset_vertex.single().unwrap();
    offset_vertex.0.iter().for_each(|c| {
        let _tess: Vec<Vec3> = c
            .tessellate(Some(tol))
            .into_iter()
            .map(|p| p.coords.cast::<f32>().to_homogeneous().into())
            .collect_vec();
        // gizmos.linestrip(tess, YELLOW);
    });
}
