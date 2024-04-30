use bevy::{
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderRef},
};

const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(16376858152701542574);

#[derive(Asset, TypePath, Default, AsBindGroup, Debug, Clone)]
pub struct LineMaterial {
    #[uniform(0)]
    pub color: Color,
}

impl Material for LineMaterial {
    fn fragment_shader() -> ShaderRef {
        ShaderRef::Handle(SHADER_HANDLE.clone())
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
