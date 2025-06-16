use bevy::{
    color::palettes::css::TOMATO,
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin, InfiniteGridSettings};

use bevy_normal_material::{plugin::NormalMaterialPlugin, prelude::NormalMaterial};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
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
        app.add_systems(Startup, setup).add_systems(Update, update);
    }
}

#[derive(Component)]
struct IntersectionSurface(pub NurbsSurface3D<f64>);

#[derive(Component)]
struct IntersectionCurve(pub NurbsCurve3D<f64>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
) {
    let profile =
        NurbsCurve3D::polyline(&[Point3::new(-2., 0., -2.), Point3::new(2., 0., -2.)], true);
    let _surface = NurbsSurface::extrude(&profile, &(Vector3::z() * 4.));

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

    let curve =
        NurbsCurve3D::try_circle(&Point3::new(0., 0., 0.), &Vector3::x(), &Vector3::y(), 1.)
            .unwrap();

    let line_vertices = curve
        .tessellate(Some(1e-8))
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, 0.])
        .collect();
    let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(line_vertices),
    );
    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: TOMATO.into(),
            ..Default::default()
        })),
        IntersectionCurve(curve.clone()),
    ));

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
            Mesh3d(
                meshes.add(PointsMesh::from_iter(intersections.iter().flat_map(|it| {
                    [
                        Vec3::from(it.a().0.cast::<f32>()),
                        Vec3::from(it.b().0.cast::<f32>()),
                    ]
                }))),
            ),
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

#[allow(clippy::type_complexity)]
fn update(
    time: Res<Time>,
    mut set: ParamSet<(
        Query<&Transform, With<Camera>>,
        Query<(&IntersectionSurface, &mut Transform), With<IntersectionSurface>>,
        Query<(&IntersectionCurve, &mut Transform), With<IntersectionCurve>>,
    )>,
    mut gizmos: Gizmos,
) {
    let speed = 1.0;
    let elapsed = time.elapsed_secs() * speed;
    let delta = time.delta_secs() * speed;

    set.p1().iter_mut().for_each(|(_, mut tr1)| {
        tr1.rotate_local_z(-delta * 0.25);
    });
    set.p2().iter_mut().for_each(|(_, mut tr2)| {
        let x = elapsed.cos() * 1.5;
        let y = elapsed.sin() * 1.5;
        *tr2 = tr2.with_translation(Vec3::new(x, y, 0.));
        tr2.rotate_local_x(delta * 1.0);
    });

    let transformed_surface = set
        .p1()
        .iter()
        .map(|(s, tr)| {
            let mat = tr.compute_matrix();
            let m = Matrix4::from(mat);
            s.0.transformed(&m.cast::<f64>())
        })
        .next()
        .unwrap();

    let transformed_curve = set
        .p2()
        .iter()
        .map(|(c2, tr)| {
            let mat = tr.compute_matrix();
            let m = Matrix4::from(mat);
            c2.0.transformed(&m.cast::<f64>())
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
    let camera_position = p0.single().unwrap().translation;

    let intersections = transformed_surface.find_intersection(&transformed_curve, None);
    if let Ok(intersections) = intersections {
        intersections.iter().for_each(|it| {
            let p: Vec3 = it.a().0.cast::<f32>().into();
            let normal = (camera_position - p).normalize();
            let _dir = Dir3::new_unchecked(normal);
            gizmos.circle(p, 1e-2 * 2.5, Color::WHITE);
        });
    }
}
