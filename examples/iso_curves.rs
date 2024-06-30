use std::f64::consts::FRAC_PI_2;

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
        view::screenshot::ScreenshotManager,
    },
    window::{close_on_esc, PrimaryWindow},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use materials::*;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};
use systems::*;

use curvo::prelude::*;
mod materials;
mod systems;

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

    let lofted = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();

    let option = AdaptiveTessellationOptions {
        norm_tolerance: 1e-2,
        ..Default::default()
    };
    let tess = lofted.tessellate(Some(option));
    let tess = tess.cast::<f32>();
    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect();

    let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices));

    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(mesh),
            material: normal_materials.add(NormalMaterial {
                opacity: 0.1,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            }),
            visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("lofted"));

    let v_direction = true;
    let (min, max) = if v_direction {
        lofted.v_knots_domain()
    } else {
        lofted.u_knots_domain()
    };
    let div = 64;
    for i in 0..div {
        let u = min + (max - min) * i as f64 / (div - 1) as f64;
        // println!("u:{}, min:{}, max:{}", u, min, max);
        let iso = lofted.try_isocurve(u, v_direction);
        match iso {
            Ok(curve) => {
                let samples = curve.tessellate(Some(1e-8));
                let line_vertices: Vec<_> = samples
                    .iter()
                    .map(|p| p.cast::<f32>())
                    .map(|p| p.into())
                    .collect();
                let n = line_vertices.len();
                let line = Mesh::new(PrimitiveTopology::LineStrip, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(line_vertices),
                    )
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_COLOR,
                        VertexAttributeValues::Float32x4(
                            (0..n)
                                .map(|i| Color::hsl(((i as f32) / n as f32) * 300., 0.5, 0.5))
                                .map(|c| c.rgba_to_vec4().into())
                                .collect(),
                        ),
                    );
                commands
                    .spawn(MaterialMeshBundle {
                        mesh: meshes.add(line),
                        material: line_materials.add(LineMaterial {
                            color: Color::WHITE,
                            ..Default::default()
                        }),
                        ..Default::default()
                    })
                    .insert(Name::new("iso curve"));

                let (start, end) = curve.knots_domain();
                let points = curve.sample_regular_range(start, end, 16);
                commands
                    .spawn(MaterialMeshBundle {
                        mesh: meshes.add(PointsMesh {
                            vertices: points.iter().map(|pt| pt.cast::<f32>().into()).collect(),
                            ..Default::default()
                        }),
                        material: points_materials.add(PointsMaterial {
                            settings: PointsShaderSettings {
                                point_size: 0.05,
                                color: Color::TOMATO,
                                ..Default::default()
                            },
                            ..Default::default()
                        }),
                        visibility: Visibility::Hidden,
                        ..Default::default()
                    })
                    .insert(Name::new("points"));
            }
            Err(e) => {
                println!("error:{}", e);
            }
        }
    }

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 3., 8.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
}
