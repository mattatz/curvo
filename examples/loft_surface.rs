use std::f64::consts::FRAC_PI_2;

use bevy::{color::palettes::css::RED, prelude::*, render::camera::ScalingMode};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use materials::*;
use misc::{add_surface, add_surface_normals};
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

use curvo::prelude::*;
mod materials;
mod misc;

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

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let add_surface_tess =
        |surf: &NurbsSurface3D<f64>,
         commands: &mut Commands<'_, '_>,
         meshes: &mut ResMut<'_, Assets<Mesh>>,
         line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
         _normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
         points_materials: &mut ResMut<'_, Assets<PointsMaterial>>| {
            let option = AdaptiveTessellationOptions::<_>::default().with_norm_tolerance(1e-2);
            let tess = surf.tessellate(Some(option));
            let tess = tess.cast::<f32>();
            // let tess = surf.regular_tessellate(32, 32);
            add_surface_normals(&tess, commands, meshes, line_materials);

            let points = surf.regular_sample_points(32, 32);
            commands
                .spawn((
                    Mesh3d(
                        meshes.add(PointsMesh {
                            vertices: points
                                .iter()
                                .flat_map(|row| row.iter().map(|pt| pt.cast::<f32>().into()))
                                .collect(),
                            ..Default::default()
                        }),
                    ),
                    MeshMaterial3d(points_materials.add(PointsMaterial {
                        settings: PointsShaderSettings {
                            point_size: 0.05,
                            color: RED.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    })),
                    Visibility::Hidden,
                ))
                .insert(Name::new("points"));
        };

    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 2.0, 0.),
        Point3::new(1.0, 2.5, 0.),
    ];
    let interpolated = NurbsCurve3D::<f64>::interpolate(&interpolation_target, 3).unwrap();

    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let translation = Translation3::new(0., 0., 1.5);
    let m = translation * rotation;
    let front = interpolated.transformed(&(translation.inverse()).into());
    let back = interpolated.transformed(&m.into());

    let lofted = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();

    add_surface::<DefaultDivider>(
        &lofted,
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        None,
    );
    add_surface_tess(
        &lofted,
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &mut normal_materials,
        &mut points_materials,
    );

    let (umin, umax) = lofted.u_knots_domain();
    let (vmin, vmax) = lofted.v_knots_domain();
    let center = lofted
        .point_at((umax + umin) * 0.5, (vmax + vmin) * 0.5)
        .cast::<f32>()
        .into();

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
        Transform::from_translation(center + Vec3::new(0., 0., 3.)).looking_at(center, Vec3::Y),
        PanOrbitCamera::default(),
    ));
}
