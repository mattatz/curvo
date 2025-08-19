use bevy::{
    color::palettes::css::{CORNFLOWER_BLUE, TOMATO, WHITE},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_normal_material::{plugin::NormalMaterialPlugin, prelude::NormalMaterial};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{
    material::PointsShaderSettings, mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial,
};
use itertools::Itertools;
use misc::surface_2_mesh;
use nalgebra::{Matrix4, Point3, Point4, Vector3};

use curvo::prelude::*;

mod materials;
mod misc;

use materials::*;
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (640., 480.).into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(NormalMaterialPlugin)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
        .add_plugins(AppPlugin)
        .run();
}
struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        app.add_systems(Startup, setup);
        //.add_systems(Update, update);
    }
}

#[derive(Component)]
struct IntersectionSurface(pub NurbsSurface3D<f64>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut standard_materials: ResMut<Assets<StandardMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let degree = 3;
    let n: usize = 6;
    let goal = n + degree + 1;
    let knots = KnotVector::uniform(goal - degree * 2, degree);
    let hn = (n - 1) as f64 / 2.;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
    let pts = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let x = i as f64 - hn;
                    let y = (rng.random::<f64>() - 0.5) * 2.;
                    let z = (j as f64) - hn;
                    Point4::new(x, y, z, 1.)
                })
                .collect_vec()
        })
        .collect_vec();
    let surface = NurbsSurface3D::new(degree, degree, knots.to_vec(), knots.to_vec(), pts);

    commands.spawn((
        Mesh3d(meshes.add(surface_2_mesh(&surface, None))),
        // Mesh3d(meshes.add(surface_2_regular_mesh(&surface, 64, 64))),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            opacity: 0.35,
            cull_mode: None,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        })),
        IntersectionSurface(surface.clone()),
    ));

    let plane = Plane::new(Vector3::y(), 0.0);
    let dir = plane.normal().cast::<f32>();
    let size = 8.;
    // ground plane
    commands.spawn((
        Mesh3d(
            meshes.add(
                Plane3d {
                    normal: Dir3::new(Vec3::from(dir)).unwrap(),
                    half_size: Vec2::new(50.0, 50.0),
                }
                .mesh()
                .size(size, size)
                .subdivisions(2),
            ),
        ),
        Transform::from_translation((dir * (plane.constant() as f32)).into()),
        MeshMaterial3d(standard_materials.add(StandardMaterial {
            base_color: Color::from(CORNFLOWER_BLUE).with_alpha(0.25),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..Default::default()
        })),
    ));

    /*
    let ta = SurfaceBoundingBoxTree::new(&surface, UVDirection::U, None);
    let leaf_nodes = ta.traverse_leaf_nodes_with_plane(&plane);
    for node in leaf_nodes {
        let s = node.surface_owned();
        commands.spawn((
            Mesh3d(meshes.add(surface_2_mesh(&s, None))),
            MeshMaterial3d(normal_materials.add(NormalMaterial {
                opacity: 0.85,
                cull_mode: None,
                alpha_mode: AlphaMode::Blend,
                ..Default::default()
            })),
        ));
    }
    */

    let its = find_surface_plane_intersection_points(&surface, &plane, None);
    if let Ok(its) = its {
        commands.spawn((
            Mesh3d(meshes.add(PointsMesh::from_iter(
                its.iter().map(|it| Vec3::from(it.0.cast::<f32>())),
            ))),
            MeshMaterial3d(points_materials.add(PointsMaterial {
                settings: PointsShaderSettings {
                    point_size: 0.025,
                    color: WHITE.into(),
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            })),
        ));
    }

    let its = surface.find_intersection(&plane, None);
    println!("its: {:?}", its);

    /*
    let ta = SurfaceBoundingBoxTree::new(&surface, UVDirection::U, None);
    let tb = CurveBoundingBoxTree::new(&circle, None);
    let traversed = BoundingBoxTraversal::try_traverse(ta, tb);
    if let Ok(traversed) = traversed {
        let n = traversed.pairs_iter().count();
        traversed
            .pairs_iter()
            .enumerate()
            .for_each(|(idx, (a, b))| {
                let t = (idx as f32) / (n as f32);
                let hue = t * 360. * 1e2 % 360.;
                let color = Color::hsl(hue, 0.5, 0.5);
                let b0 = a.bounding_box();
                let b1 = b.bounding_box();
                let vertices = b0
                    .lines()
                    .iter()
                    .chain(b1.lines().iter())
                    .flat_map(|(a, b)| {
                        [a, b]
                            .iter()
                            .map(|p| p.cast::<f32>())
                            .map(|p| [p.x, p.y, 0.])
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let line = Mesh::new(PrimitiveTopology::LineList, default())
                    .with_inserted_attribute(
                        Mesh::ATTRIBUTE_POSITION,
                        VertexAttributeValues::Float32x3(vertices),
                    );
                commands
                    .spawn((
                        Mesh3d(meshes.add(line)),
                        MeshMaterial3d(line_materials.add(LineMaterial {
                            // color: Color::WHITE,
                            color,
                            opacity: 0.5,
                            alpha_mode: AlphaMode::Blend,
                        })),
                    ))
                    .insert(Name::new("bounding box"));
            });
    }
    */

    /*
    let intersections = surface.find_intersection(&circle, None);
    if let Ok(intersections) = intersections {
        commands.spawn((
        ));
    }
    */

    let center = Vec3::ZERO;
    commands.spawn((
        Transform::from_translation(center + Vec3::new(3., 3., 3.)).looking_at(center, Vec3::Y),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle {
        settings: InfiniteGridSettings {
            x_axis_color: Color::BLACK,
            z_axis_color: Color::BLACK,
            ..Default::default()
        },
        ..Default::default()
    });
}
