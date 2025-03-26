use bevy::{
    color::palettes::css::TOMATO,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{material::NormalMaterial, plugin::NormalMaterialPlugin};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{Point2, Point3, Point4, Vector2, Vector3};

use curvo::prelude::*;
use rand::Rng;
use systems::screenshot_on_spacebar;
mod materials;
mod misc;
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
            .add_systems(Update, screenshot_on_spacebar);
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    _points_materials: ResMut<Assets<PointsMaterial>>,
    mut normal_materials: ResMut<'_, Assets<NormalMaterial>>,
) {
    let add_trimmed_surface = |commands: &mut Commands<'_, '_>,
                               meshes: &mut ResMut<'_, Assets<Mesh>>,
                               normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
                               surface: &TrimmedSurface<f64>,
                               transform: Transform| {
        let option = AdaptiveTessellationOptions {
            norm_tolerance: 1e-2,
            ..Default::default()
        };
        let tess = surface.tessellate(Some(option)).unwrap().cast::<f32>();
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
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(normal_materials.add(NormalMaterial {
                cull_mode: None,
                ..Default::default()
            })),
            transform,
        ));
    };

    let add_surface = |commands: &mut Commands<'_, '_>,
                       meshes: &mut ResMut<'_, Assets<Mesh>>,
                       normal_materials: &mut ResMut<'_, Assets<NormalMaterial>>,
                       surface: &NurbsSurface3D<f64>,
                       transform: Transform| {
        let option = AdaptiveTessellationOptions {
            norm_tolerance: 1e-2,
            ..Default::default()
        };
        let tess = surface.tessellate(Some(option)).cast::<f32>();
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
        commands.spawn((
            Mesh3d(meshes.add(mesh)),
            MeshMaterial3d(normal_materials.add(NormalMaterial {
                cull_mode: None,
                ..Default::default()
            })),
            transform,
        ));
    };

    let add_curve = |commands: &mut Commands<'_, '_>,
                     meshes: &mut ResMut<'_, Assets<Mesh>>,
                     line_materials: &mut ResMut<'_, Assets<LineMaterial>>,
                     curve: &NurbsCurve3D<f64>,
                     transform: Transform| {
        let (min, max) = curve.knots_domain();
        let interval = max - min;
        let step = interval / 32.;
        let mut t = min;
        let mut vertices: Vec<Vec3> = vec![];

        while t <= max {
            let pt = curve.point_at(t).cast::<f32>();
            vertices.push(pt.into());

            t += step;
        }

        let samples = curve.tessellate(Some(1e-8));
        let mut line = Mesh::new(bevy::render::mesh::PrimitiveTopology::LineStrip, default());
        let line_vertices = samples
            .iter()
            .map(|p| p.cast::<f32>())
            .map(|p| [p.x, p.y, p.z])
            .collect();
        line.insert_attribute(
            Mesh::ATTRIBUTE_POSITION,
            VertexAttributeValues::Float32x3(line_vertices),
        );
        commands.spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: TOMATO.into(),
                ..Default::default()
            })),
            transform,
        ));
    };

    let plane_surface = NurbsSurface::plane(
        Point3::new(2.5, 0., 5.),
        Vector3::x() * 2.5,
        Vector3::z() * 5.,
    );
    let trimmed = TrimmedSurface::new(
        plane_surface.clone(),
        None,
        vec![
            NurbsCurve2D::try_circle(&Point2::new(0.5, 0.5), &Vector2::x(), &Vector2::y(), 0.25)
                .unwrap(),
        ],
    );

    let offset = 5.0;

    let trimming_curve = NurbsCurve3D::try_circle(
        &Point3::new(2.5, 5.0, 2.5),
        &Vector3::x(),
        &Vector3::z(),
        1.,
    )
    .unwrap();
    let trimmed2 = TrimmedSurface::try_projection(
        plane_surface.clone(),
        -Vector3::y(),
        Some(trimming_curve.clone()),
        vec![],
    )
    .unwrap();

    add_trimmed_surface(
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        &trimmed,
        Transform::default(),
    );

    let transform = Transform::from_translation(Vec3::new(offset, 0., 0.));
    add_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &trimming_curve,
        transform,
    );
    add_trimmed_surface(
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        &trimmed2,
        transform,
    );

    let transform = Transform::from_translation(Vec3::new(offset * 2., 0., 0.));
    let degree = 3;
    let n: usize = 6;
    let goal = n + degree + 1;
    let knots = KnotVector::uniform(goal - degree * 2, degree);
    let _hn = (n - 1) as f64 / 2.;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
    let pts = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let x = i as f64;
                    let z = (rng.random::<f64>() - 0.5) * 2.;
                    let y = j as f64;
                    Point4::new(x, z, y, 1.)
                })
                .collect_vec()
        })
        .collect_vec();
    let wavy_surface = NurbsSurface3D::new(degree, degree, knots.to_vec(), knots.to_vec(), pts);

    let trimmed3 = TrimmedSurface::try_projection(
        wavy_surface.clone(),
        -Vector3::y(),
        Some(trimming_curve.clone()),
        vec![],
    )
    .unwrap();
    add_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &trimming_curve,
        transform,
    );
    add_surface(
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        &wavy_surface,
        transform,
    );
    add_trimmed_surface(
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        &trimmed3,
        transform * Transform::from_translation(Vec3::new(0., 1.5, 0.)),
    );

    commands.spawn((
        Transform::from_translation(Vec3::new(18., 18., 18.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
