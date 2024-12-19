use std::f32::consts::FRAC_PI_2;

use bevy::prelude::*;
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};

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
        app.add_systems(Startup, setup);
    }
}

#[derive(Component)]
struct FirstCurve(pub NurbsCurve2D<f64>);

#[derive(Component)]
struct SecondCurve(pub NurbsCurve2D<f64>);

fn setup(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let center = Vec3::ZERO;
    commands.spawn((
        Transform::from_translation(center + Vec3::new(0., 0., 8.)).looking_at(center, Vec3::Y),
        PanOrbitCamera::default(),
    ));
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

/*
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
*/
