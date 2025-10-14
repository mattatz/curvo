use bevy::{
    color::palettes::css::{BLUE, CORNFLOWER_BLUE, GREEN, RED, TOMATO, YELLOW},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
};

use bevy_normal_material::{plugin::NormalMaterialPlugin, prelude::NormalMaterial};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use nalgebra::{Point3, Point4, Translation3, Vector3, U4};

use curvo::prelude::*;

mod materials;
mod misc;

use materials::*;
use misc::*;
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
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

const N: usize = 6;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
    mut points_materials: ResMut<Assets<PointsMaterial>>,
    mut mesh_materials: ResMut<Assets<StandardMaterial>>,
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
) {
    // Create reference surface (plane at z=0)
    let ref_surface =
        NurbsSurface3D::<f64>::plane(Point3::origin(), Vector3::x() * 2.0, Vector3::y() * 2.0);

    /*
    // Create target surface (sphere)
    let target_surface = NurbsSurface3D::try_sphere(
        &Point3::new(0.0, 0.0, 3.0),
        &Vector3::z(),
        &Vector3::x(),
        1.5,
    )
    .unwrap();
    */

    let degree = 3;
    let goal = N + degree + 1;
    let knots = KnotVector::uniform(goal - degree * 2, degree);
    let hn = (N - 1) as f64 / 2.;
    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([0; 32]);
    let pts = (0..N)
        .map(|i| {
            (0..N)
                .map(|j| {
                    let x = i as f64 - hn;
                    let y = (rng.random::<f64>() - 0.5) * 2.;
                    let z = (j as f64) - hn;
                    Point4::new(x, y, z, 1.)
                })
                .collect_vec()
        })
        .collect_vec();
    let target_surface = NurbsSurface3D::new(degree, degree, knots.to_vec(), knots.to_vec(), pts);
    let target_surface =
        target_surface.transformed(&nalgebra::convert(Translation3::new(0., 0., 5.)));

    render_surface(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &ref_surface,
        CORNFLOWER_BLUE.into(),
        0.1,
    );
    render_surface(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &target_surface,
        CORNFLOWER_BLUE.into(),
        0.1,
    );

    // Example 1: Morph a point
    let resolution = 10;
    let s = 4.;
    let inv_resolution = 1.0 / resolution as f64;
    let pts = (0..=resolution)
        .flat_map(|x| {
            let fx = x as f64 * inv_resolution - 0.5;
            (0..=resolution).map(move |y| {
                let fy = y as f64 * inv_resolution - 0.5;
                let point = Point3::new(fx * s, fy * s, 0.0);
                point
            })
        })
        .collect_vec();

    // Render original and morphed point
    let point_size = 0.025;
    render_point(
        &mut commands,
        &mut meshes,
        &mut points_materials,
        &pts,
        YELLOW.into(),
        point_size,
    );

    render_point(
        &mut commands,
        &mut meshes,
        &mut points_materials,
        &pts.iter().map(|p| p.morph(&ref_surface, &target_surface).unwrap()).collect_vec(),
        YELLOW.into(),
        point_size,
    );

    // Example 2: Morph a curve
    let curve_points = vec![
        Point3::new(-1.25, -1.25, 0.0),
        Point3::new(0.0, -1.25, 0.0),
        Point3::new(1.25, -1.25, 0.0),
        Point3::new(1.25, 0.0, 0.0),
        Point3::new(1.25, 1.25, 0.0),
    ];
    let curve = NurbsCurve3D::interpolate(&curve_points, 1).unwrap();
    let morphed_curve = curve.morph(&ref_surface, &target_surface).unwrap();

    [curve, morphed_curve].into_iter().for_each(|curve| {
        render_curve(
            &mut commands,
            &mut meshes,
            &mut line_materials,
            &curve,
            TOMATO.into(),
        );
    });

    let circle = NurbsCurve3D::try_circle(
        &Point3::new(-0.5, 0.5, 0.),
        &Vector3::x(),
        &Vector3::y(),
        1.25,
    )
    .unwrap();
    let morphed_circle = circle.morph(&ref_surface, &target_surface).unwrap();
    [circle, morphed_circle].into_iter().for_each(|circle| {
        render_curve(
            &mut commands,
            &mut meshes,
            &mut line_materials,
            &circle,
            TOMATO.into(),
        );
    });

    // Setup camera
    let scale = 8.0;
    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scale,
            near: 1e-1,
            far: 1e4,
            scaling_mode: ScalingMode::FixedVertical {
                viewport_height: 1.,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_translation(Vec3::new(5., 5., 8.))
            .looking_at(Vec3::new(0., 0., 1.5), Vec3::Y),
        PanOrbitCamera::default(),
    ));
}

fn render_surface(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    line_materials: &mut ResMut<Assets<LineMaterial>>,
    surface: &NurbsSurface3D<f64>,
    color: Color,
    alpha: f32,
) {
    let divs = 64;
    let (u_domain, v_domain) = surface.knots_domain();

    // Create wireframe mesh
    let mut vertices = Vec::new();

    // U direction lines
    for i in 0..=divs {
        let u = u_domain.0 + (u_domain.1 - u_domain.0) * (i as f64 / divs as f64);
        for j in 0..divs {
            let v0 = v_domain.0 + (v_domain.1 - v_domain.0) * (j as f64 / divs as f64);
            let v1 = v_domain.0 + (v_domain.1 - v_domain.0) * ((j + 1) as f64 / divs as f64);

            let p0 = surface.point_at(u, v0);
            let p1 = surface.point_at(u, v1);

            vertices.push([p0.x as f32, p0.y as f32, p0.z as f32]);
            vertices.push([p1.x as f32, p1.y as f32, p1.z as f32]);
        }
    }

    // V direction lines
    for j in 0..=divs {
        let v = v_domain.0 + (v_domain.1 - v_domain.0) * (j as f64 / divs as f64);
        for i in 0..divs {
            let u0 = u_domain.0 + (u_domain.1 - u_domain.0) * (i as f64 / divs as f64);
            let u1 = u_domain.0 + (u_domain.1 - u_domain.0) * ((i + 1) as f64 / divs as f64);

            let p0 = surface.point_at(u0, v);
            let p1 = surface.point_at(u1, v);

            vertices.push([p0.x as f32, p0.y as f32, p0.z as f32]);
            vertices.push([p1.x as f32, p1.y as f32, p1.z as f32]);
        }
    }

    let line = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );

    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: color.with_alpha(alpha),
            opacity: alpha,
            alpha_mode: AlphaMode::Blend,
            ..Default::default()
        })),
    ));
}

fn render_curve(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    line_materials: &mut ResMut<Assets<LineMaterial>>,
    curve: &NurbsCurve3D<f64>,
    color: Color,
) {
    let vertices = curve
        .tessellate(Some(1e-8))
        .iter()
        .map(|p| p.cast::<f32>())
        .map(|p| [p.x, p.y, p.z])
        .collect();

    let line = Mesh::new(PrimitiveTopology::LineStrip, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );

    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color,
            ..Default::default()
        })),
    ));
}

fn render_point(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    points_materials: &mut ResMut<Assets<PointsMaterial>>,
    points: &[Point3<f64>],
    color: Color,
    size: f32,
) {
    commands.spawn((
        Mesh3d(meshes.add(PointsMesh {
            vertices: points.iter().map(|p| p.cast::<f32>().into()).collect(),
            ..Default::default()
        })),
        MeshMaterial3d(points_materials.add(PointsMaterial {
            settings: bevy_points::material::PointsShaderSettings {
                color: color.into(),
                point_size: size,
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        })),
    ));
}
