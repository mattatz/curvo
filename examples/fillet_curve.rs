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
use nalgebra::{Point2, Point3};

use curvo::prelude::*;

mod materials;
mod misc;

use materials::*;
use misc::*;
use rand::Rng;

#[derive(Debug, PartialEq)]
enum Dimension {
    Two,
    Three,
}

impl Default for Dimension {
    fn default() -> Self {
        Self::Two
    }
}

#[derive(Resource, Debug)]
struct Setting {
    pub radius: f64,
    pub dimension: Dimension,
    pub control_points: bool,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            radius: 0.2,
            dimension: Dimension::default(),
            control_points: true,
        }
    }
}

#[derive(Component)]
struct ControlPoints(pub Vec<Point2<f64>>);

#[derive(Component)]
struct ProfileCurve(pub NurbsCurve2D<f64>);

#[derive(Component)]
struct Profile3DCurve(pub NurbsCurve3D<f64>);

#[derive(Component)]
struct FilletCurve(pub CompoundCurve2D<f64>);

#[derive(Component)]
struct Fillet3DCurve(pub CompoundCurve3D<f64>);

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
            .add_systems(Update, gizmos_curve);
    }
}

fn setup(mut commands: Commands, settings: Res<Setting>) {
    // s-shape
    let _points = [
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 1.0),
        Point2::new(-1.0, 1.0),
        Point2::new(-1.0, 3.0),
        Point2::new(1.0, 3.0),
    ];

    // square
    let points = vec![
        Point2::new(-1.0, -1.0),
        Point2::new(1.0, -1.0),
        Point2::new(1.0, 1.0),
        Point2::new(-1.0, 1.0),
        // Point2::new(-1.0, -1.0),
    ];

    // convex polygon
    let points = vec![
        Point2::new(-0.5, 2.),
        Point2::new(0.5, 2.),
        Point2::new(0.5, 1.),
        Point2::new(1.5, 1.),
        Point2::new(1.5, -1.),
        Point2::new(-1.5, -1.),
        Point2::new(-1.5, 1.),
        Point2::new(-0.5, 1.),
        Point2::new(-0.5, 2.),
    ]
    .into_iter()
    .rev()
    .collect_vec();

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
    let delta = 0.5;
    let points = points
        .into_iter()
        .map(|p| {
            let dx = (rng.random::<f64>() - 0.5) * delta;
            let dy = (rng.random::<f64>() - 0.5) * delta;
            Point2::new(p.x + dx, p.y + dy)
        })
        .collect_vec();

    commands.spawn((ControlPoints(points.clone()),));

    let curve = NurbsCurve2D::polyline(&points, true);
    let option = FilletRadiusOption::new(settings.radius);
    let fillet_curve = curve.fillet(option.clone()).unwrap();

    commands.spawn((ProfileCurve(curve),));
    commands.spawn((FilletCurve(fillet_curve),));

    let curve = NurbsCurve3D::polyline(
        &points
            .iter()
            .map(|p| {
                let p: Point3<f64> = p.coords.to_homogeneous().into();
                let dz = (rng.random::<f64>() - 0.5) * delta;
                Point3::new(p.x, p.y, p.z + dz)
                // Point3::new(p.x, p.y, p.z)
            })
            .collect_vec(),
        true,
    );
    let option = FilletRadiusOption::new(settings.radius);
    let fillet_curve = curve.fillet(option.clone()).unwrap();

    commands.spawn((Profile3DCurve(curve),));
    commands.spawn((Fillet3DCurve(fillet_curve),));

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
            // orbit_sensitivity: 0.0,
            ..Default::default()
        },
    ));
}

fn update_ui(
    mut contexts: EguiContexts,
    mut settings: ResMut<Setting>,
    profile: Query<&ProfileCurve>,
    profile_3d: Query<&Profile3DCurve>,
    mut fillet_curve: Query<&mut FilletCurve>,
    mut fillet_curve_3d: Query<&mut Fillet3DCurve>,
) {
    let mut changed = false;
    egui::Window::new("fillet curve")
        .collapsible(false)
        .drag_to_scroll(false)
        .default_width(420.)
        .min_width(420.)
        .max_width(420.)
        .show(contexts.ctx_mut(), |ui| {
            ui.heading("dimension");
            ui.group(|g| {
                g.horizontal(|ui| {
                    ui.selectable_value(&mut settings.dimension, Dimension::Two, "2D");
                    ui.selectable_value(&mut settings.dimension, Dimension::Three, "3D");
                });
            });

            ui.heading("radius");
            ui.group(|g| {
                g.horizontal(|ui| {
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut settings.radius)
                                .speed(1e-2)
                                .range(0.0..=5.0),
                        )
                        .changed();
                });
            });

            ui.heading("control points");
            ui.group(|g| {
                g.horizontal(|ui| {
                    ui.checkbox(&mut settings.control_points, "show");
                });
            });
        });

    if changed {
        let profile = profile.single().unwrap();

        let mut fillet_curve = fillet_curve.single_mut().unwrap();
        let option = FilletRadiusOption::new(settings.radius);
        let res = profile.0.fillet(option);
        if let Ok(res) = res {
            fillet_curve.0 = res;
        }

        let profile_3d = profile_3d.single().unwrap();
        let mut fillet_curve_3d = fillet_curve_3d.single_mut().unwrap();
        let res = profile_3d.0.fillet(option);
        if let Ok(res) = res {
            fillet_curve_3d.0 = res;
        }
    }
}

fn gizmos_curve(
    control_points: Query<&ControlPoints>,
    profile: Query<&ProfileCurve>,
    fillet_curve: Query<&FilletCurve>,
    fillet_curve_3d: Query<&Fillet3DCurve>,
    settings: Res<Setting>,
    mut gizmos: Gizmos,
) {
    let points = &control_points.single().unwrap().0;
    points.iter().for_each(|p| {
        let p: Vec3 = p.coords.cast::<f32>().to_homogeneous().into();
        // gizmos.sphere(p, 0.025, WHITE);
    });

    let tol = 1e-7 * 0.5;

    let profile = profile.single().unwrap();
    let tess = profile
        .0
        .tessellate(Some(tol))
        .into_iter()
        .map(|p| p.coords.cast::<f32>().to_homogeneous().into())
        .collect_vec();
    gizmos.linestrip(tess, WHITE.with_alpha(0.25));

    match settings.dimension {
        Dimension::Two => {
            let fillet_curve = fillet_curve.single().unwrap();
            fillet_curve.0.spans().iter().for_each(|c| {
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

                if settings.control_points {
                    c.dehomogenized_control_points().iter().for_each(|p| {
                        let p: Vec3 = p.coords.cast::<f32>().to_homogeneous().into();
                        gizmos.sphere(p, 0.025, LIGHT_GREEN);
                    });
                }
            });
        }
        Dimension::Three => {
            let fillet_curve = fillet_curve_3d.single().unwrap();
            fillet_curve.0.spans().iter().for_each(|c| {
                let tess: Vec<Vec3> = c
                    .tessellate(Some(tol))
                    .into_iter()
                    .map(|p| p.coords.cast::<f32>().into())
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

                if settings.control_points {
                    c.dehomogenized_control_points().iter().for_each(|p| {
                        let p: Vec3 = p.coords.cast::<f32>().into();
                        gizmos.sphere(p, 0.025, LIGHT_GREEN);
                    });
                }
            });
        }
    }
}
