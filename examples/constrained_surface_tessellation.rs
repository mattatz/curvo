use std::f64::consts::FRAC_PI_2;

use bevy::{
    color::palettes::css::TOMATO,
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    },
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use itertools::Itertools;
use materials::*;
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

    let (umin, umax) = front.knots_domain();

    let surface = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();

    let u_parameters = (0..=8)
        .map(|i| umin + (umax - umin) * i as f64 / 8.)
        .collect_vec();
    let vmin = surface.v_knots_domain().0;

    let boundary = BoundaryConstraints::default().with_u_parameters_at_v_min(u_parameters.clone());

    commands.spawn((
        Mesh3d(
            meshes.add(PointsMesh {
                vertices: u_parameters
                    .iter()
                    .map(|u| surface.point_at(*u, vmin).cast::<f32>().into())
                    .collect_vec(),
                ..Default::default()
            }),
        ),
        MeshMaterial3d(points_materials.add(PointsMaterial {
            settings: PointsShaderSettings {
                point_size: 0.05,
                color: TOMATO.into(),
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        })),
    ));

    // let tess = surface.tessellate(Some(Default::default()));
    let option = AdaptiveTessellationOptions::<_> {
        ..Default::default()
    };
    let tess = surface.constrained_tessellate(boundary, Some(option));
    let tess = tess.cast::<f32>();

    let vertices = tess.points().iter().map(|pt| (*pt).into()).collect_vec();
    commands.spawn((
        Mesh3d(meshes.add(PointsMesh {
            vertices: vertices.clone(),
            ..Default::default()
        })),
        MeshMaterial3d(points_materials.add(PointsMaterial {
            settings: PointsShaderSettings {
                point_size: 0.02,
                color: Color::WHITE.into(),
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        })),
    ));

    let normals = tess.normals().iter().map(|n| (*n).into()).collect();
    let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
    let indices = tess
        .faces()
        .iter()
        .flat_map(|f| f.iter().map(|i| *i as u32))
        .collect_vec();

    let mesh = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        // triangle edges
        VertexAttributeValues::Float32x3(
            indices
                .chunks(3)
                .flat_map(|idx| {
                    [
                        vertices[idx[0] as usize].into(),
                        vertices[idx[1] as usize].into(),
                        vertices[idx[1] as usize].into(),
                        vertices[idx[2] as usize].into(),
                        vertices[idx[2] as usize].into(),
                        vertices[idx[0] as usize].into(),
                    ]
                })
                .collect_vec(),
        ),
    );
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: Color::WHITE,
            ..Default::default()
        })),
    ));

    let mesh = Mesh::new(PrimitiveTopology::TriangleList, default())
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(vertices.into_iter().map(|v| v.into()).collect()),
        )
        .with_inserted_attribute(
            Mesh::ATTRIBUTE_NORMAL,
            VertexAttributeValues::Float32x3(normals),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs))
        .with_inserted_indices(Indices::U32(indices));

    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            cull_mode: None,
            ..Default::default()
        })),
    ));

    let (umin, umax) = surface.u_knots_domain();
    let (vmin, vmax) = surface.v_knots_domain();
    let center = surface
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
