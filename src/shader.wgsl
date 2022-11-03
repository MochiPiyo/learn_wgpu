//vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    //@location(1) color: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
};

struct VertexOutput {
    //@builtin(position)はclip corrdinates
    @builtin(position) clip_position: vec4<f32>,
    //@location(0) color: vec3<f32>
    @location(0) tex_coords: vec2<f32>,
};

//valid entry point for a vertex shader
@vertex
fn vs_main(
    model: VertexInput,
    //rust側でrender_pass.draw(0..3, 0..1);とかする
    //@builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
    //var is valiable but specify type. let can have inferred type, but cannot change value
    var out: VertexOutput;
    //out.color = model.color;
    out.tex_coords = model.tex_coords;
    
    //let x = f32(1 - i32(in_vertex_index)) * 0.5;
    //let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    //out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    //out.clip_position = vec4<f32>(model.position, 1.0);

    //order is important for matrix. vector is right, matrices is left
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    return out;
}

//Fragment shader
//@group はBindGroupLayoutとBindGroupを作るときに関係する
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

//fragment shader entry point
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //this sets the color of current fragment to brown
    //return vec4<f32>(0.3, 0.2, 0.1, 1.0);
    //return vec4<f32>(in.color, 1.0);
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}