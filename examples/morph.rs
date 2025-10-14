use bevy::{
    color::palettes::css::{BLUE, GREEN, RED, YELLOW},
    prelude::*,
    render::{
        camera::ScalingMode,
        mesh::{PrimitiveTopology, VertexAttributeValues},
    },
};

use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use nalgebra::{Point3, Point4, Translation3, Vector3};

use curvo::prelude::*;

mod materials;

use materials::*;
use rand::Rng;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(PointsPlugin)
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

    // Render reference surface (transparent wireframe)
    render_surface(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &ref_surface,
        Color::from(GREEN),
        "Reference Surface (Plane)",
        0.5,
    );

    // Render target surface (transparent wireframe)
    render_surface(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &target_surface,
        Color::from(BLUE),
        "Target Surface (Sphere)",
        0.5,
    );

    // Example 1: Morph a point
    let point = Point3::new(0.5, 0.5, 0.0);
    let morphed_point = point.morph(&ref_surface, &target_surface).unwrap();

    // Render original and morphed point
    render_point(
        &mut commands,
        &mut meshes,
        &mut points_materials,
        &point,
        Color::from(RED),
        "Original Point",
        0.1,
    );
    render_point(
        &mut commands,
        &mut meshes,
        &mut points_materials,
        &morphed_point,
        Color::from(YELLOW),
        "Morphed Point",
        0.1,
    );

    // Example 2: Morph a curve
    let curve_points = vec![
        Point3::new(-1.0, -1.0, 0.0),
        Point3::new(0.0, -1.0, 0.0),
        Point3::new(1.0, -1.0, 0.0),
        Point3::new(1.0, 0.0, 0.0),
        Point3::new(1.0, 1.0, 0.0),
    ];
    let curve = NurbsCurve3D::try_interpolate(&curve_points, 3).unwrap();
    let morphed_curve = curve.morph(&ref_surface, &target_surface).unwrap();

    // Render original and morphed curve
    render_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &curve,
        Color::from(RED),
        "Original Curve",
    );
    render_curve(
        &mut commands,
        &mut meshes,
        &mut line_materials,
        &morphed_curve,
        Color::from(YELLOW),
        "Morphed Curve",
    );

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
    name: &str,
    alpha: f32,
) {
    let divs = 20;
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

    commands
        .spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color: color.with_alpha(alpha),
                ..Default::default()
            })),
        ))
        .insert(Name::new(name.to_string()));
}

fn render_curve(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    line_materials: &mut ResMut<Assets<LineMaterial>>,
    curve: &NurbsCurve3D<f64>,
    color: Color,
    name: &str,
) {
    let samples = 100;
    let (start, end) = curve.knots_domain();
    let mut vertices = Vec::new();

    for i in 0..samples {
        let t0 = start + (end - start) * (i as f64 / samples as f64);
        let t1 = start + (end - start) * ((i + 1) as f64 / samples as f64);

        let p0 = curve.point_at(t0);
        let p1 = curve.point_at(t1);

        vertices.push([p0.x as f32, p0.y as f32, p0.z as f32]);
        vertices.push([p1.x as f32, p1.y as f32, p1.z as f32]);
    }

    let line = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(vertices),
    );

    commands
        .spawn((
            Mesh3d(meshes.add(line)),
            MeshMaterial3d(line_materials.add(LineMaterial {
                color,
                ..Default::default()
            })),
        ))
        .insert(Name::new(name.to_string()));
}

fn render_point(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    points_materials: &mut ResMut<Assets<PointsMaterial>>,
    point: &Point3<f64>,
    color: Color,
    name: &str,
    size: f32,
) {
    commands
        .spawn((
            Mesh3d(meshes.add(PointsMesh {
                vertices: vec![Vec3::new(point.x as f32, point.y as f32, point.z as f32)],
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
        ))
        .insert(Name::new(name.to_string()));
}
