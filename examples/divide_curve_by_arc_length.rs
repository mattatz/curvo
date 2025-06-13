use bevy::{
    color::palettes::css::{ALICE_BLUE, AQUAMARINE, WHITE, YELLOW},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::plugin::PointsPlugin;
use materials::*;
use nalgebra::Point2;

use curvo::prelude::*;
mod materials;

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
        .add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: true,
        })
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

#[derive(Resource, Default)]
struct Setting {
    pub arc_length: f64,
    pub min_arc_length: f64,
    pub max_arc_length: f64,
    pub parameters: Vec<CurveLengthParameter<f64>>,
}

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
            .add_systems(EguiContextPass, (update_ui))
            .add_systems(Update, (divide_by_arc_length));
    }
}

#[derive(Component)]
struct ProfileCurve(pub NurbsCurve2D<f64>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut settings: ResMut<Setting>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    let control_points: Vec<Point2<f64>> = vec![
        Point2::new(-1., -1.),
        Point2::new(1., -1.),
        Point2::new(1., 0.),
        Point2::new(-1., 0.),
        Point2::new(-1., 1.),
        Point2::new(1., 1.),
    ];
    let degree = 3;
    let curve = NurbsCurve2D::try_interpolate(&control_points, degree).unwrap();

    let length = curve.try_length().unwrap();
    settings.arc_length = length / 5.;
    settings.min_arc_length = length / 100.;
    settings.max_arc_length = length;
    let divided = curve.try_divide_by_length(settings.arc_length);
    if let Ok(divided) = divided {
        settings.parameters = divided;
    }

    let vertices = curve
        .cast::<f32>()
        .tessellate(Some(1e-3))
        .iter()
        .map(|p| [p.x, p.y, 0.])
        .collect();
    let mesh = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );

    commands.spawn((
        ProfileCurve(curve),
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: WHITE.into(),
            ..Default::default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}

fn absorb_egui_inputs(mut mouse: ResMut<ButtonInput<MouseButton>>, mut contexts: EguiContexts) {
    if contexts.ctx_mut().is_pointer_over_area() {
        mouse.reset_all();
    }
}

fn update_ui(
    mut contexts: EguiContexts,
    profile: Query<&ProfileCurve>,
    mut settings: ResMut<Setting>,
) {
    let profile = profile.single().unwrap();
    egui::Window::new("divide curve by arc length example")
        .collapsible(false)
        .drag_to_scroll(false)
        .default_width(420.)
        .min_width(420.)
        .max_width(420.)
        .show(contexts.ctx_mut(), |ui| {
            let range = settings.min_arc_length..=settings.max_arc_length;
            let response = ui.add(
                egui::Slider::new(&mut settings.arc_length, range)
                    .logarithmic(true)
                    .text("arc length"),
            );
            if response.changed() {
                let divided = profile.0.try_divide_by_length(settings.arc_length);
                if let Ok(divided) = divided {
                    settings.parameters = divided;
                }
            }
        });
}

fn divide_by_arc_length(profile: Query<&ProfileCurve>, settings: Res<Setting>, mut gizmos: Gizmos) {
    let profile = profile.single().unwrap();
    let range = settings.max_arc_length - settings.min_arc_length;
    let r = ((settings.arc_length / range).min(1e-1)) as f32;
    let p3d = profile.0.elevate_dimension();
    let frames = p3d.compute_frenet_frames(
        &settings
            .parameters
            .iter()
            .map(|p| p.parameter())
            .collect::<Vec<_>>(),
    );
    frames.iter().for_each(|f| {
        let pt = f.position();
        let tangent = f.tangent() * 1e-1;
        let binormal = f.binormal() * 1e-1;
        gizmos.circle(
            Isometry3d::from_translation(pt.cast()),
            // Dir3::new_unchecked(Vec3::from(normal)),
            r,
            ALICE_BLUE,
        );
        gizmos.line(
            pt.cast::<f32>().into(),
            (pt + tangent).cast::<f32>().into(),
            AQUAMARINE,
        );
        gizmos.line(
            pt.cast::<f32>().into(),
            (pt + binormal).cast::<f32>().into(),
            YELLOW,
        );
    });
}
