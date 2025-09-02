use std::f64::consts::FRAC_PI_2;

use bevy::{prelude::*, render::mesh::PrimitiveTopology};
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{plugin::NormalMaterialPlugin, prelude::NormalMaterial};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use materials::*;
use misc::surface_2_mesh;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

use curvo::prelude::*;
mod materials;
mod misc;

#[derive(Resource)]
struct Setting {
    pub direction: UVDirection,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            direction: UVDirection::U,
        }
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
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(NormalMaterialPlugin)
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
            .add_systems(EguiContextPass, update_ui)
            .add_systems(Update, split_animation);
    }
}

#[derive(Component)]
struct ProfileSurface(pub NurbsSurface3D<f64>);

#[derive(Component)]
struct FirstSurface;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
) {
    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 2.0, 0.),
        Point3::new(1.0, 2.5, 0.),
    ];
    let interpolated = NurbsCurve3D::<f64>::try_interpolate(&interpolation_target, 3).unwrap();

    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let translation = Translation3::new(0., 0., 1.5);
    let m = translation * rotation;
    let front = interpolated.transformed(&(translation.inverse()).into());
    let back = interpolated.transformed(&m.into());

    let surface = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();
    commands.spawn(ProfileSurface(surface));

    commands.spawn((
        FirstSurface,
        Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::TriangleList, default()))),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            cull_mode: None,
            ..Default::default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(5., 5., 5.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}

fn split_animation(
    time: Res<Time>,
    profile: Query<&ProfileSurface>,
    mut meshes: ResMut<Assets<Mesh>>,
    first: Query<&Mesh3d, With<FirstSurface>>,
    setting: Res<Setting>,
) {
    let profile = profile.single().unwrap();
    let direction = setting.direction;
    let (u, v) = profile.0.knots_domain();
    let sec = time.elapsed_secs_f64() * 2.;
    let rng = match direction {
        UVDirection::U => u,
        UVDirection::V => v,
    };
    let t = rng.0 + (rng.1 - rng.0) * (0.5 + 0.5 * sec.sin());
    let option = SplitSurfaceOption::new(t, direction);
    let (s0, _s1) = profile.0.try_split(option).unwrap();
    let first = first.single().unwrap();
    let mesh = surface_2_mesh(
        &s0,
        Some(AdaptiveTessellationOptions::<_> {
            max_depth: 4,
            ..Default::default()
        }),
    );
    *meshes.get_mut(first).unwrap() = mesh;
}

fn update_ui(mut contexts: EguiContexts, mut settings: ResMut<Setting>) {
    egui::Window::new("Split surface example")
        .collapsible(false)
        .drag_to_scroll(false)
        .default_width(420.)
        .min_width(420.)
        .max_width(420.)
        .show(contexts.ctx_mut(), |ui| {
            egui::ComboBox::from_label("split direction")
                .selected_text(format!("{:?}", settings.direction))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut settings.direction, UVDirection::U, "U");
                    ui.selectable_value(&mut settings.direction, UVDirection::V, "V");
                });
        });
}
