use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderRef, ShaderType},
};

const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(16376858152701542574);

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
            color: Color::rgb(1.0, 1.0, 1.0),
            opacity: 1.0,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

#[derive(ShaderType)]
struct LineMaterialUniform {
    color: Color,
    opacity: f32,
    #[cfg(feature = "webgl")]
    _webgl2_padding: bevy::math::Vec3,
}

impl From<&LineMaterial> for LineMaterialUniform {
    fn from(material: &LineMaterial) -> LineMaterialUniform {
        LineMaterialUniform {
            color: material.color,
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

    fn alpha_mode(&self) -> bevy::prelude::AlphaMode {
        self.alpha_mode
    }

    fn depth_bias(&self) -> f32 {
        0.0
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline<Self>,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::render::mesh::MeshVertexBufferLayout,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        // WebGL2 structs must be 16 byte aligned.
        let shader_defs = vec![
            #[cfg(feature = "webgl")]
            "SIXTEEN_BYTE_ALIGNMENT".into(),
        ];
        if let Some(fragment) = &mut descriptor.fragment {
            fragment.shader_defs = shader_defs;
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

use bevy::{
    asset::load_internal_asset,
    prelude::{MaterialPlugin, Plugin, Shader},
};

pub struct LineMaterialPlugin;

impl Plugin for LineMaterialPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../shaders/line_material.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(MaterialPlugin::<LineMaterial>::default());
    }
}
