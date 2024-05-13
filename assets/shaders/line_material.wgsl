#import bevy_pbr::forward_io::VertexOutput

struct LineMaterial {
    color: vec4<f32>,
    opacity: f32,
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding: vec3<f32>
#endif
};

@group(2) @binding(0) var<uniform> material: LineMaterial;

@fragment
fn fragment(
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    vec4 c = material.color;
    c.a *= material.opacity;
    return c;
}
