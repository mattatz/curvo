use std::f64::consts::FRAC_PI_2;

use bevy::{
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_egui::{egui, EguiContextPass, EguiContexts, EguiPlugin};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};

use bevy_normal_material::{plugin::NormalMaterialPlugin, prelude::NormalMaterial};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use itertools::Itertools;
use materials::*;
use misc::surface_2_mesh;
use nalgebra::{Point3, Rotation3, Translation3, Vector2, Vector3, U4};

use curvo::prelude::*;

use crate::misc::add_surface;
mod materials;
mod misc;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (640., 480.).into(),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .add_plugins(InfiniteGridPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(NormalMaterialPlugin)
        .add_plugins(LineMaterialPlugin)
        .add_plugins(EguiPlugin {
            enable_multipass_for_primary_context: true,
        })
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
    mut normal_materials: ResMut<Assets<NormalMaterial>>,
    mut line_materials: ResMut<Assets<LineMaterial>>,
) {
    let field = |uv: Vector2<f64>| {
        let center = Vector2::new(0.5, 0.5);
        let radius = 0.2;
        let thickness = 0.1;
        let distance = (uv - center).norm();
        if (radius - distance).abs() <= thickness {
            Some(DividableDirection::Both)
        } else {
            None
        }
    };

    let surface = NurbsSurface::plane(Point3::new(0., 0., 0.), Vector3::x() * 2., Vector3::y());
    let options: AdaptiveTessellationOptions<f64, U4, _> = AdaptiveTessellationOptions {
        min_depth: 3,
        max_depth: 12,
        divider: Some(|node: &AdaptiveTessellationNode<f64, U4>| {
            let corners = node.corners().iter().map(|c| field(*c.uv())).collect_vec();
            let uv = node.uv_center();
            let center = field(uv);
            // Some(DividableDirection::Both)
            if corners[0] != corners[1]
                || corners[2] != corners[3]
                || corners[0] != corners[2]
                || corners[1] != corners[3]
                || center != corners[0]
                || center != corners[1]
                || center != corners[2]
                || center != corners[3]
            {
                Some(DividableDirection::Both)
            } else {
                None
            }
        }),
        ..Default::default()
    };
    add_surface(
        &surface,
        &mut commands,
        &mut meshes,
        &mut normal_materials,
        Some(options.clone()),
    );

    let tess = surface.tessellate(Some(options));
    let vertices: Vec<[f32; 3]> = tess
        .points()
        .iter()
        .map(|pt| pt.cast::<f32>().into())
        .collect_vec();
    let line = Mesh::new(PrimitiveTopology::LineList, default()).with_inserted_attribute(
        Mesh::ATTRIBUTE_POSITION,
        VertexAttributeValues::Float32x3(
            tess.faces()
                .iter()
                .map(|f| {
                    let pt = f.iter().map(|i| vertices[*i]).collect_vec();
                    [pt[0], pt[1], pt[1], pt[2], pt[2], pt[0]]
                })
                .flatten()
                .collect_vec(),
        ),
    );
    commands.spawn((
        Mesh3d(meshes.add(line)),
        MeshMaterial3d(line_materials.add(LineMaterial {
            color: Color::WHITE,
            ..Default::default()
        })),
    ));

    /*
    commands.spawn((
        Mesh3d(meshes.add(Mesh::new(PrimitiveTopology::TriangleList, default()))),
        MeshMaterial3d(normal_materials.add(NormalMaterial {
            cull_mode: None,
            ..Default::default()
        })),
    ));
    */

    commands.spawn((
        Transform::from_translation(Vec3::new(5., 5., 5.)),
        PanOrbitCamera::default(),
    ));
    commands.spawn(InfiniteGridBundle::default());
}
