use std::ops::Range;

use bevy::{
    color::palettes::css::WHITE,
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{material::PointsMaterial, mesh::PointsMesh, plugin::PointsPlugin};
use easer::functions::Easing;
use materials::*;
use nalgebra::{Point3, Point4};

use curvo::prelude::*;
use rand_distr::{Distribution, UnitSphere};
mod materials;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
        .add_plugins(NormalMaterialPlugin)
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

#[derive(Resource)]
struct Setting {
    pub duration: f32,
    pub time_since_last_update: f32,
    pub degrees: Range<usize>,
    pub num_control_points: usize,
    pub samples: usize,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            duration: 5.,
            time_since_last_update: 0.,
            degrees: (1..4),
            num_control_points: 8,
            samples: 36,
        }
    }
}

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.insert_resource(ClearColor(Color::srgb(0.97, 0.39, 0.62)))
            .insert_resource(Setting::default())
            .add_systems(Startup, setup)
            .add_systems(Update, (animate, reset));
    }
}

#[derive(Component)]
struct SourceControlPoints;

#[derive(Component)]
struct ProfileCurve(pub NurbsCurve3D<f32>);

fn setup(mut commands: Commands, mut settings: ResMut<Setting>) {
    settings.time_since_last_update = settings.duration;

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 0., 3.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
    // commands.spawn(InfiniteGridBundle::default());
}

fn animate(
    mut gizmos: Gizmos,
    time: Res<Time>,
    mut source: Query<&mut Transform, With<SourceControlPoints>>,
    profile: Query<&ProfileCurve>,
    mut settings: ResMut<Setting>,
) {
    let delta = time.delta_secs();

    let t = settings.time_since_last_update / settings.duration;
    let t = (t * (1.0 + 1e-2)).min(1.0); // add padding
    let st = easer::functions::Expo::ease_in(t, 0., 1., 1.);
    let et = easer::functions::Expo::ease_out(t, 0., 1., 1.);

    let tr = source.get_single_mut();
    let tr = if let Ok(mut tr) = tr {
        tr.rotate_local_y(delta * 0.25);
        *tr
    } else {
        Transform::IDENTITY
    };

    profile.iter().for_each(|profile| {
        let curve = &profile.0;
        let (start, end) = curve.knots_domain();
        let interval = end - start;

        // redefine the knot range
        let (start, end) = (start + interval * st, start + interval * et);
        let interval = end - start;
        let is_most_accurate = curve.degree() >= settings.degrees.end - 1;

        let samples = if is_most_accurate {
            settings.samples * 2
        } else {
            settings.samples
        };
        let inv = 1.0 / ((samples - 1) as f32);
        let pts = (0..samples)
            .map(|i| {
                let u = (i as f32) * inv * interval + start;
                let p = curve.point_at(u);
                let pt = Vec3::from(p);
                tr.transform_point(pt)
            })
            .collect::<Vec<_>>();

        if is_most_accurate {
            [pts.first(), pts.last()].iter().for_each(|pt| {
                gizmos.sphere(*pt.unwrap(), 1e-2, Color::BLACK);
            });
            gizmos.linestrip(pts.iter().copied(), Color::hsla(0.0, 0.0, 0.0, 1.));
        } else {
            pts.iter().for_each(|pt| {
                gizmos.sphere(*pt, 1e-2 * 0.5, Color::hsla(0., 0., 0., 0.5));
            });
            gizmos.linestrip(pts.iter().copied(), Color::hsla(0.0, 0.0, 0.0, 0.35));
        }
    });

    settings.time_since_last_update += delta;
}

#[allow(clippy::too_many_arguments)]
fn reset(
    mut commands: Commands,
    _time: Res<Time>,
    source: Query<Entity, With<SourceControlPoints>>,
    profile: Query<Entity, With<ProfileCurve>>,
    mut settings: ResMut<Setting>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    if settings.time_since_last_update > settings.duration {
        settings.time_since_last_update = 0.;

        source.iter().for_each(|e| {
            commands.entity(e).despawn_recursive();
        });
        profile.iter().for_each(|e| {
            commands.entity(e).despawn_recursive();
        });

        let pts: Vec<Point3<f32>> = (0..settings.num_control_points)
            .map(|_i| {
                let sample: [f32; 3] = UnitSphere.sample(&mut rand::thread_rng());
                Point3::from_slice(&sample)
            })
            .collect();

        commands
            .spawn((
                SourceControlPoints,
                Mesh3d(meshes.add(PointsMesh {
                    vertices: pts.iter().map(|pt| (*pt).into()).collect(),
                    colors: None,
                })),
                MeshMaterial3d(points_materials.add(PointsMaterial {
                    settings: bevy_points::material::PointsShaderSettings {
                        color: WHITE.into(),
                        point_size: 0.025,
                        opacity: 0.5,
                    },
                    alpha_mode: AlphaMode::Blend,
                    circle: true,
                    ..Default::default()
                })),
            ))
            .with_children(|f| {
                f.spawn((
                    Mesh3d(meshes.add(
                        Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
                            Mesh::ATTRIBUTE_POSITION,
                            VertexAttributeValues::Float32x3(
                                pts.iter().map(|pt| [pt.x, pt.y, pt.z]).collect(),
                            ),
                        ),
                    )),
                    MeshMaterial3d(line_materials.add(LineMaterial {
                        color: Color::WHITE,
                        opacity: 0.5,
                        alpha_mode: AlphaMode::Blend,
                    })),
                ));
            });

        let homogenized: Vec<Point4<f32>> =
            pts.iter().map(|pt| pt.to_homogeneous().into()).collect();
        settings.degrees.clone().for_each(|degree| {
            let goal = homogenized.len() + degree + 1;
            let knots = KnotVector::uniform(goal - degree * 2, degree).to_vec();
            let _ = NurbsCurve::try_new(degree, homogenized.clone(), knots).map(|curve| {
                let curve = ProfileCurve(curve);
                commands.spawn((curve,));
            });
        });
    }
}
