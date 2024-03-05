use wgpu::SurfaceConfiguration;
use winit::event::{WindowEvent, KeyboardInput, ElementState, VirtualKeyCode};

use cgmath;

pub struct Camera {
    pub eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}
#[rustfmt::skip]
//wgpu is based on DirectX and Metal. cgmath crate is built for OpenGL, so convert x axis and y axis
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);
impl Camera {
    pub fn new(config: &SurfaceConfiguration) -> Self {
        Self {
            //positon the camera one unit up and 2 units back
            //+z is tou of the screen
            eye: (0.0, 1.0, 2.0).into(),
            //have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            //which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    pub fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        //camera
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        //presentate depth, without this, far and close object are same size
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        //
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}


pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_back_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_back_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }
    
    pub fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        return true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_back_pressed = is_pressed;
                        return true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        return true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        return true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        //Prebents glitching when camera gets too close to the center of the scene
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_back_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);
        //Redo radius calc in case the forward/back is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            //Rescale the distance between the target and eye so that it doesn't change.
            //The eye therefore still lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }

    }
}