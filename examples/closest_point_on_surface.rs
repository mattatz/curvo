use std::f64::consts::FRAC_PI_2;

use bevy::{
    color::palettes::css::WHITE,
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    },
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{Point3, Point4, Rotation3, Translation3, Vector3};

use curvo::prelude::*;
use rand::Rng;
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

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, find_closest_point);
    }
}

#[derive(Component)]
struct TargetSurface(pub NurbsSurface3D<f64>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let add_surface =
        |surf: &NurbsSurface3D<f64>,
         commands: &mut Commands<'_, '_>,
         meshes: &mut ResMut<'_, Assets<Mesh>>,
         line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
         normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
         points_materials: &mut ResMut<'_, Assets<PointsMaterial>>| {
            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());

            let option = AdaptiveTessellationOptions {
                norm_tolerance: 1e-2,
                ..Default::default()
            };
            let tess = surf.tessellate(Some(option));
            let tess = tess.cast::<f32>();

            let mut line_list =
                Mesh::new(bevy::render::mesh::PrimitiveTopology::LineList, default());
            let normal_length = 0.15;
            let normals = tess.normals();

            let vertices = tess
                .points()
                .iter()
                .enumerate()
                .flat_map(|(i, p)| {
                    let pt: Vec3 = (*p).into();
                    let normal: Vec3 = normals[i].normalize().into();
                    [pt, pt + normal * normal_length]
                })
                .map(|p| p.to_array())
                .collect();

            line_list.insert_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(vertices),
            );

            let vertices = tess.points().iter().map(|pt| (*pt).into()).collect();
            let normals = tess.normals().iter().map(|n| (*n).into()).collect();
            let uvs = tess.uvs().iter().map(|uv| (*uv).into()).collect();
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
                .insert(Name::new("surface"));
        };

    // let surface = NurbsSurface3D::try_sphere(Point3::origin(), Vector3::z(), Vector3::x(), 1.).unwrap();
    let degree = 3;
    let n = 6;
    let goal = n + degree + 1;
    let knots = KnotVector::uniform(goal - degree * 2, degree);
    let hn = (n - 1) as f64 / 2.;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
    let pts = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let x = i as f64 - hn;
                    let z = (rng.gen::<f64>() - 0.5) * 2.;
                    let y = j as f64 - hn;
                    Point4::new(x, z, y, 1.)
                })
                .collect_vec()
        })
        .collect_vec();
    let surface = NurbsSurface3D::new(degree, degree, knots.to_vec(), knots.to_vec(), pts);

    add_surface(
        &surface,
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &mut normal_materials,
        &mut points_materials,
    );

    let camera = Camera3dBundle {
        transform: Transform::from_translation(Vec3::new(0., 2.5, 10.)),
        ..Default::default()
    };
    commands.spawn((camera, PanOrbitCamera::default()));
    commands.spawn(InfiniteGridBundle::default());

    let u_div = 10;
    let v_div = 10;
    let nf = (n - 1) as f64;
    let pts = (0..=u_div)
        .flat_map(|iu| {
            let fu = iu as f64 / u_div as f64 * nf - hn;
            (0..=v_div)
                .map(|iv| {
                    let fv = iv as f64 / v_div as f64 * nf - hn;
                    let pt = Point3::new(fu, 2., fv);
                    pt
                })
                .collect_vec()
        })
        .collect_vec();

    commands
        .spawn(MaterialMeshBundle {
            mesh: meshes.add(PointsMesh {
                vertices: pts.iter().map(|p| p.cast::<f32>().coords.into()).collect(),
                ..Default::default()
            }),
            material: points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: WHITE.into(),
                    point_size: 0.05,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            }),
            // visibility: Visibility::Hidden,
            ..Default::default()
        })
        .insert(Name::new("points"));

    pts.iter().for_each(|pt| {
        if let Ok(closest) = surface.find_closest_point(&pt) {
            let line_vertices = [pt, &closest]
                .iter()
                .map(|p| p.cast::<f32>().into())
                .collect_vec();
            let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
                Mesh::ATTRIBUTE_POSITION,
                VertexAttributeValues::Float32x3(line_vertices),
            );
            commands.spawn(MaterialMeshBundle {
                mesh: meshes.add(line),
                material: line_materials.add(LineMaterial {
                    color: WHITE.with_alpha(0.25).into(),
                    opacity: 0.6,
                    alpha_mode: AlphaMode::Blend,
                    ..Default::default()
                }),
                ..Default::default()
            });
        } else {
            println!("Failed to find closest point");
        }
    });

    commands.spawn(TargetSurface(surface));
}

fn find_closest_point(time: Res<Time>, surfaces: Query<&TargetSurface>, mut gizmos: Gizmos) {
    /*
    let t = time.elapsed_seconds_f64();
    let point = Point3::new(t.sin(), t.cos(), 0.) * 5.;
    gizmos.sphere(point.cast::<f32>().into(), Quat::IDENTITY, 0.25, WHITE);
    surfaces.iter().for_each(|s| {
        let closest = s.0.find_closest_point(&point);
        if let Ok(closest) = closest {
            gizmos.line(
                point.cast::<f32>().into(),
                closest.cast::<f32>().into(),
                WHITE,
            );
        }
    });
    */
}
