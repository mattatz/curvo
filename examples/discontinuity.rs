use bevy::{
    color::palettes::css::{TOMATO, WHITE, YELLOW},
    prelude::*,
    render::mesh::{PrimitiveTopology, VertexAttributeValues},
};
use bevy_infinite_grid::InfiniteGridPlugin;

use bevy_normal_material::plugin::NormalMaterialPlugin;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use bevy_points::{mesh::PointsMesh, plugin::PointsPlugin, prelude::PointsMaterial};
use itertools::Itertools;
use materials::*;
use nalgebra::{Point3, Point4};

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
        Point3::new(131.226349, -7.9162e-7, 115.987156),
        Point3::new(131.226351, 0.836042, 115.987156),
        Point3::new(131.001564, 1.672085, 115.987156),
        Point3::new(127.835282, 7.14587, 115.987156),
        Point3::new(123.051333, 16.898002, 115.987156),
        Point3::new(118.117737, 31.970953, 115.987156),
        Point3::new(115.432967, 47.494182, 115.987156),
        Point3::new(115.114209, 58.139094, 115.987156),
        Point3::new(115.360456, 64.402558, 115.987156),
        Point3::new(114.888823, 66.330993, 115.987156),
        Point3::new(113.455076, 67.704089, 115.987156),
        Point3::new(107.907208, 70.6218, 115.987156),
        Point3::new(98.847811, 76.220739, 115.987156),
        Point3::new(86.748383, 86.30719, 115.987156),
        Point3::new(77.924489, 96.148338, 115.987156),
        Point3::new(71.623756, 104.881473, 115.987156),
        Point3::new(68.72489, 109.535048, 115.987156),
        Point3::new(67.04496, 112.442853, 115.987156),
        Point3::new(66.214787, 113.39027, 115.987156),
        Point3::new(64.673363, 114.151043, 115.987156),
        Point3::new(63.659824, 114.283692, 115.987156),
        Point3::new(57.727832, 114.287277, 115.987156),
        Point3::new(46.891136, 115.01433, 115.987156),
        Point3::new(31.370345, 118.280239, 115.987156),
        Point3::new(16.585103, 123.715094, 115.987156),
        Point3::new(7.206987, 128.761856, 115.987156),
        Point3::new(1.905702, 132.106772, 115.987156),
        Point3::new(7.0197e-8, 132.662919, 115.987156),
        Point3::new(-1.905701, 132.106772, 115.987156),
        Point3::new(-7.206988, 128.761868, 115.987156),
        Point3::new(-16.585094, 123.715102, 115.987156),
        Point3::new(-31.370355, 118.280328, 115.987156),
        Point3::new(-46.891096, 115.014404, 115.987156),
        Point3::new(-57.727895, 114.287604, 115.987156),
        Point3::new(-63.659392, 114.283226, 115.987156),
        Point3::new(-64.877627, 114.126541, 115.987156),
        Point3::new(-66.39751, 113.249129, 115.987156),
        Point3::new(-67.14224, 112.272335, 115.987156),
        Point3::new(-70.112364, 107.138479, 115.987156),
        Point3::new(-79.182173, 93.603892, 115.987156),
        Point3::new(-94.316121, 79.016115, 115.987156),
        Point3::new(-107.907694, 70.627503, 115.987156),
        Point3::new(-113.067542, 67.904042, 115.987156),
        Point3::new(-113.812277, 67.363941, 115.987156),
        Point3::new(-115.005638, 65.845204, 115.987156),
        Point3::new(-115.360358, 64.403364, 115.987156),
        Point3::new(-115.218658, 60.834196, 115.987156),
        Point3::new(-115.199117, 55.476851, 115.987156),
        Point3::new(-115.879995, 44.906899, 115.987156),
        Point3::new(-119.017155, 26.798172, 115.987156),
        Point3::new(-125.06902, 11.947654, 115.987156),
        Point3::new(-131.000214, 1.672045, 115.987156),
        Point3::new(-131.449297, 0.000012, 115.987156),
        Point3::new(-131.000204, -1.672012, 115.987156),
        Point3::new(-125.06926, -11.947676, 115.987156),
        Point3::new(-119.01571, -26.797765, 115.987156),
        Point3::new(-115.432859, -47.493959, 115.987156),
        Point3::new(-115.118294, -58.136932, 115.987156),
        Point3::new(-115.340484, -63.967379, 115.987156),
        Point3::new(-115.245034, -64.882395, 115.987156),
        Point3::new(-114.526456, -66.675199, 115.987156),
        Point3::new(-113.45519, -67.703393, 115.987156),
        Point3::new(-110.293184, -69.365054, 115.987156),
        Point3::new(-105.64422, -72.027296, 115.987156),
        Point3::new(-92.42338, -80.838271, 115.987156),
        Point3::new(-79.18322, -93.604955, 115.987156),
        Point3::new(-70.112164, -107.138038, 115.987156),
        Point3::new(-67.142793, -112.273102, 115.987156),
        Point3::new(-66.5212, -113.084594, 115.987156),
        Point3::new(-65.091598, -114.039068, 115.987156),
        Point3::new(-63.856054, -114.284344, 115.987156),
        Point3::new(-60.497811, -114.285248, 115.987156),
        Point3::new(-55.018566, -114.469236, 115.987156),
        Point3::new(-44.304673, -115.558699, 115.987156),
        Point3::new(-31.371939, -118.281887, 115.987156),
        Point3::new(-19.049319, -122.808335, 115.987156),
        Point3::new(-9.555225, -127.503817, 115.987156),
        Point3::new(-4.925331, -130.199354, 115.987156),
        Point3::new(-1.905217, -132.106675, 115.987156),
        Point3::new(-0.479178, -132.52039, 115.987156),
        Point3::new(1.432761, -132.246283, 115.987156),
        Point3::new(2.272872, -131.871372, 115.987156),
        Point3::new(7.21138, -128.764544, 115.987156),
        Point3::new(21.271905, -121.188205, 115.987156),
        Point3::new(41.472261, -115.375709, 115.987156),
        Point3::new(57.728449, -114.288305, 115.987156),
        Point3::new(63.659559, -114.283027, 115.987156),
        Point3::new(64.877852, -114.126469, 115.987156),
        Point3::new(66.397654, -113.248917, 115.987156),
        Point3::new(67.142465, -112.272237, 115.987156),
        Point3::new(70.111996, -107.137588, 115.987156),
        Point3::new(76.15983, -98.11606, 115.987156),
        Point3::new(86.748472, -86.307502, 115.987156),
        Point3::new(98.847834, -76.22064, 115.987156),
        Point3::new(107.907228, -70.621851, 115.987156),
        Point3::new(113.455089, -67.704105, 115.987156),
        Point3::new(114.888836, -66.331014, 115.987156),
        Point3::new(115.360469, -64.402575, 115.987156),
        Point3::new(115.11422, -58.13911, 115.987156),
        Point3::new(115.432972, -47.494189, 115.987156),
        Point3::new(118.11774, -31.970958, 115.987156),
        Point3::new(123.051334, -16.898004, 115.987156),
        Point3::new(127.835282, -7.145872, 115.987156),
        Point3::new(131.001564, -1.672087, 115.987156),
        Point3::new(131.226351, -0.836044, 115.987156),
        Point3::new(131.226349, -7.9162e-7, 115.987156),
    ];
    let knots = vec![
        0.0, 0.0, 0.0, 0.0, 0.002744, 0.002744, 0.022109, 0.041474, 0.060839, 0.080204, 0.080204,
        0.083333, 0.086463, 0.086463, 0.105828, 0.125193, 0.144558, 0.15424, 0.163923, 0.163923,
        0.165981, 0.168039, 0.16941, 0.16941, 0.188775, 0.20814, 0.227506, 0.246871, 0.246871,
        0.25, 0.253129, 0.253129, 0.272494, 0.29186, 0.311225, 0.33059, 0.33059, 0.331961,
        0.334705, 0.336077, 0.336077, 0.355442, 0.394172, 0.413537, 0.413537, 0.415102, 0.416667,
        0.419796, 0.419796, 0.429479, 0.439161, 0.458526, 0.497256, 0.497256, 0.5, 0.502744,
        0.502744, 0.541474, 0.560839, 0.580204, 0.580204, 0.581769, 0.583333, 0.586463, 0.586463,
        0.596145, 0.605828, 0.644558, 0.663923, 0.663923, 0.665295, 0.667353, 0.66941, 0.66941,
        0.679093, 0.688775, 0.70814, 0.727506, 0.737188, 0.746871, 0.746871, 0.75, 0.751565,
        0.753129, 0.753129, 0.772494, 0.811225, 0.83059, 0.83059, 0.831961, 0.834705, 0.836077,
        0.836077, 0.855442, 0.874807, 0.894172, 0.913537, 0.913537, 0.916667, 0.919796, 0.919796,
        0.939161, 0.958526, 0.977891, 0.997256, 0.997256, 1.0, 1.0, 1.0, 1.0,
    ];
    let curve = NurbsCurve::try_new(
        3,
        pts.iter().map(|p| Point4::new(p.x, p.y, p.z, 1.)).collect(),
        knots,
    )
    .unwrap();

    let bb = BoundingBox::from_iter(curve.dehomogenized_control_points());

    let mut mesh = Mesh::new(PrimitiveTopology::LineStrip, default());
    let vertices = curve
        .cast::<f32>()
        .tessellate(Some(1e-6))
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
                    color: TOMATO.into(),
                    point_size: 2.0,
                    ..Default::default()
                },
                circle: true,
                ..Default::default()
            })),
        ))
        .insert(Name::new("discontinuities"));

    commands.spawn((
        Mesh3d(
            meshes.add(PointsMesh {
                vertices: [t0, t1]
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
                color: YELLOW.into(),
                point_size: 2.0,
                ..Default::default()
            },
            circle: true,
            ..Default::default()
        })),
    ));

    commands.spawn((
        Transform::from_translation(Vec3::new(0., 0., 5.)),
        PanOrbitCamera {
            focus: bb.center().cast::<f32>().into(),
            radius: Some(bb.size().cast::<f32>().norm()),
            ..Default::default()
        },
    ));
    // commands.spawn(InfiniteGridBundle::default());
}
