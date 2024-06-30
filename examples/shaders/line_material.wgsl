#import bevy_pbr::forward_io::VertexOutput

struct LineMaterial {
    color: vec4<f32>,
};

@group(2) @binding(0) var<uniform> material: LineMaterial;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    #ifdef VERTEX_COLORS
        return mesh.color;
    #else
        return material.color;
    #endif
}
