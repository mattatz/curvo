use bevy::{
    asset::uuid_handle,
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};

const SHADER_HANDLE: Handle<Shader> = uuid_handle!("558d5700-88d3-405b-aa87-82f925828be3");

#[derive(AsBindGroup, Asset, TypePath, Debug, Clone)]
#[bind_group_data(LineMaterialKey)]
#[uniform(0, LineMaterialUniform)]
pub struct LineMaterial {
    pub color: Color,
    pub opacity: f32,
    pub alpha_mode: AlphaMode,
}

impl Default for LineMaterial {
    fn default() -> Self {
        Self {
            color: Color::srgb(1.0, 1.0, 1.0),
            opacity: 1.0,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

#[derive(Clone, Default, ShaderType)]
struct LineMaterialUniform {
    color: Vec4,
    opacity: f32,
    #[cfg(feature = "webgl")]
    _webgl2_padding: bevy::math::Vec3,
}

impl From<&LineMaterial> for LineMaterialUniform {
    fn from(material: &LineMaterial) -> LineMaterialUniform {
        LineMaterialUniform {
            color: LinearRgba::from(material.color).to_f32_array().into(),
            opacity: material.opacity,
            #[cfg(feature = "webgl")]
            _webgl2_padding: Default::default(),
        }
    }
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(SHADER_HANDLE.clone())
    }

    fn alpha_mode(&self) -> AlphaMode {
        self.alpha_mode
    }

    fn depth_bias(&self) -> f32 {
        0.0
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        #[cfg(feature = "webgl")]
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());
        }

        if layout.0.contains(Mesh::ATTRIBUTE_COLOR) {
            if let Some(fragment) = &mut descriptor.fragment {
                fragment.shader_defs.push("VERTEX_COLORS".into());
            }
        }

        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct LineMaterialKey {}

impl From<&LineMaterial> for LineMaterialKey {
    fn from(_material: &LineMaterial) -> Self {
        LineMaterialKey {}
    }
}

pub struct LineMaterialPlugin;

impl Plugin for LineMaterialPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        // Use Shader::from_wgsl_direct to bypass naga_oil preprocessing,
        // then bevy's pipeline will process #{MATERIAL_BIND_GROUP} at pipeline creation time.
        let shader_source = include_str!("shaders/line_material.wgsl");
        app.world_mut()
            .resource_mut::<Assets<Shader>>()
            .insert(&SHADER_HANDLE, Shader::from_wgsl(shader_source, file!()));
        app.add_plugins(MaterialPlugin::<LineMaterial>::default());
    }
}
