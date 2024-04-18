use std::f64::consts::FRAC_PI_2;

use bevy::{
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{Indices, MeshVertexBufferLayout, PrimitiveTopology, VertexAttributeValues},
        render_resource::{
            AsBindGroup, PolygonMode, RenderPipelineDescriptor, ShaderRef,
            SpecializedMeshPipelineError,
        },
    },
    window::close_on_esc,
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use bevy_polyline::prelude::*;
use nalgebra::{Point3, Rotation3, Translation3, Vector3};

use curvo::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MaterialPlugin::<LineMaterial>::default())
        .add_plugins(PolylinePlugin)
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
            .add_systems(Update, close_on_esc);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _polylines: ResMut<Assets<Polyline>>,
    _polyline_materials: ResMut<Assets<PolylineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let add_surface =
        |surf: &NurbsSurface3D<f64>,
         commands: &mut Commands<'_, '_>,
         meshes: &mut ResMut<'_, Assets<Mesh>>,
         line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
         normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
         _points_materials: &mut ResMut<'_, Assets<PointsMaterial>>| {
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());

            let option = AdaptiveTessellationOptions {
                norm_tolerance: 1e-2,
                ..Default::default()
            };
            let tess = surf.tessellate(Some(option));
            // let tess = surf.tessellate(None);

            let mut line_list =
                Mesh::new(bevy::render::mesh::PrimitiveTopology::LineList, default());
            let normal_length = 0.15;
            let normals = tess.normals();

            let vertices = tess
                .points()
                .iter()
                .enumerate()
                .flat_map(|(i, p)| {
                    let pt: Vec3 = p.cast::<f32>().into();
                    let normal: Vec3 = normals[i].cast::<f32>().normalize().into();
                    [pt, pt + normal * normal_length]
                })
                .map(|p| p.to_array())
                .collect();

            line_list.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(vertices),
            );

            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(line_list),
                    material: line_materials.add(LineMaterial {
                        color: Color::YELLOW,
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("normal"));

            /*
            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(PointsMesh {
                        vertices: tess
                            .points()
                            .iter()
                            .map(|pt| pt.cast::<f32>().into())
                            .collect(),
                        ..Default::default()
                    }),
                    material: points_materials.add(PointsMaterial {
                        settings: PointsShaderSettings {
                            point_size: 0.05,
                            color: Color::RED,
                            ..Default::default()
                        },
                        ..Default::default()
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("points"));
            */

            let vertices = tess
                .points()
                .iter()
                .map(|pt| pt.cast::<f32>().into())
                .collect();
            let normals = tess
                .normals()
                .iter()
                .map(|n| n.cast::<f32>().into())
                .collect();
            let uvs = tess
                .uvs()
                .iter()
                .map(|uv| uv.cast::<f32>().into())
                .collect();
            let indices = tess
                .faces()
                .iter()
                .flat_map(|f| f.iter().map(|i| *i as u32))
                .collect();

            mesh.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(vertices),
            );
            mesh.insert_attribute(
                Mesh::ATTRIBUTE_NORMAL,
                VertexAttributeValues::Float32x3(normals),
            );
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, VertexAttributeValues::Float32x2(uvs));
            mesh.insert_indices(Indices::U32(indices));

            commands
                .spawn(MaterialMeshBundle {
                    mesh: meshes.add(mesh),
                    material: normal_materials.add(NormalMaterial {
                        cull_mode: None,
                        ..Default::default()
                    }),
                    // visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(Name::new("lofted"));
        };

    let interpolation_target = vec![
        Point3::new(-1.0, -1.0, 0.),
        Point3::new(1.0, -1.0, 0.),
        Point3::new(1.0, 1.0, 0.),
        Point3::new(-1.0, 1.0, 0.),
        Point3::new(-1.0, 2.0, 0.),
        Point3::new(1.0, 2.5, 0.),
    ];
    let interpolated =
        NurbsCurve3D::<f64>::try_interpolate(&interpolation_target, 3, None, None).unwrap();

    let rotation = Rotation3::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let translation = Translation3::new(0., 0., 1.5);
    let m = rotation * translation;
    let front = interpolated.transformed(&(translation.inverse()).into());
    let back = interpolated.transformed(&m.into());

    let lofted = NurbsSurface::try_loft(&[front, back], Some(3)).unwrap();

    add_surface(
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
    let orth = Camera3dBundle {
        projection: OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical(2.0),
            ..Default::default()
        }
        .into(),
        transform: Transform::from_translation(center + Vec3::new(0., 0., 3.))
            .looking_at(center, Vec3::Y),
        ..Default::default()
    };
    commands.spawn((orth, PanOrbitCamera::default()));
}

#[derive(Asset, TypePath, Default, AsBindGroup, Debug, Clone)]
struct LineMaterial {
    #[uniform(0)]
    color: Color,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/line_material.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayout,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // This is the important part to tell bevy to render this material as a line between vertices
        descriptor.primitive.polygon_mode = PolygonMode::Line;
        Ok(())
    }
}
