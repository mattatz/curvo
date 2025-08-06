use bevy::{
    color::palettes::css::{CORNFLOWER_BLUE, GOLD, WHITE},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::Point3;

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
) {
    let pts = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.3, 0.0, 0.0),
        Point3::new(0.8, 0.8, 0.0),
        Point3::new(1.2, 1.6, 0.0),
        Point3::new(1.8, 1.6, 0.0),
        Point3::new(2.2, 0.8, 0.0),
        Point3::new(2.7, 0.1, 0.0),
        Point3::new(3.0, 0.0, 0.0),
    ]
    .into_iter()
    .map(|p| p.to_homogeneous().into())
    .collect_vec();

    let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0];
    let curve = NurbsCurve3D::try_new(3, pts, knots).unwrap();

    let bb = BoundingBox::from_iter(curve.dehomogenized_control_points());

    let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, default());
    let vertices = curve
        .cast::<f32>()
        .tessellate(Some(1e-7))
        .iter()
        .map(|p| [p.x, p.y, p.z])
        .collect();
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );
    commands
        .spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: WHITE.into(),
                ..Default::default()
            })),
        ))
        .insert(Name::new("curve"));

    let (t0, t1) = curve.knots_domain();
    let discontinuities = curve
        .discontinuity_iter(DiscontinuityType::G2, t0, t1)
        .collect_vec();
    println!("{:?}", discontinuities.len());

    commands
        .spawn((
            Mesh3d(
                meshes.add(PointsMesh {
                    vertices: discontinuities
                        .iter()
                        .map(|t| {
                            let p = curve.point_at(*t).cast::<f32>();
                            [p.x, p.y, p.z].into()
                        })
                        .collect(),
                    ..Default::default()
                }),
            ),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: bevy_points::material::PointsShaderSettings {
                    color: CORNFLOWER_BLUE.into(),
                    point_size: 0.05,
                    opacity: 0.75,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            })),
            // Visibility::Hidden,
        ))
        .insert(Name::new("discontinuities"));

    let (start, end) = curve.knots_domain();
    let samples = 256;
    let span = (end - start) / ((samples - 1) as f64);
    let parameters: Vec<_> = (0..samples).map(|i| start + span * (i as f64)).collect();
    let curvatures = parameters
        .iter()
        .map(|parameter| {
            let p = curve.point_at(*parameter).cast::<f32>();
            let c = curve.curvature_at(*parameter).unwrap().cast::<f32>();
            (p, c)
        })
        .collect_vec();

    let mut normals = vec![];

    let length = 0.15;
    curvatures.iter().for_each(|(p, c)| {
        let p: Vec3 = (*p).into();
        let cv: Vec3 = c.curvature_vector().into();
        normals.push(p);
        normals.push(p + cv * length);
    });

    let add_arrows = |commands: &mut Commands<'_, '_>,
                      meshes: &mut ResMut<'_, Assets<Mesh>>,
                      line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
                      vs: &Vec<Vec3>,
                      color: Color,
                      name: String| {
        commands
            .spawn((
                Mesh3d(meshes.add(
                    Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(vs.iter().map(|v| v.to_array()).collect()),
                    ),
                )),
                MeshMaterial3d(line_materials.add(LineMaterial {
                    color,
                    opacity: 0.5,
                    alpha_mode: AlphaMode::Blend,
                })),
            ))
            .insert(Name::new(name));
    };
    add_arrows(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &normals,
        GOLD.into(),
        "n".to_string(),
    );

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 5.)),
        PanOrbitCamera {
            focus: bb.center().cast::<f32>().into(),
            radius: Some(bb.size().cast::<f32>().norm()),
            ..Default::default()
        },
    ));
    commands.spawn(InfiniteGridBundle::default());
}
