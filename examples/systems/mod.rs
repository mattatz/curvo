use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    window::PrimaryWindow,
};

pub fn screenshot_on_spacebar(
    mut commands: Commands,
    input: Res<ButtonInput<KeyCode>>,
    mut counter: Local<u32>,
) {
    if input.just_pressed(KeyCode::Space) {
        let path = format!("./screenshot-{}.png", *counter);
        *counter += 1;
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
}
