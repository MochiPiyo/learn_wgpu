//vertex shader

struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> camera: CameraUniform;

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
};
@group(2) @binding(0)
var<uniform> light: Light;

struct VertexInput {
    @location(0) position: vec3<f32>,
    //@location(1) color: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,

    //light
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
};

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    //light
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
}

struct VertexOutput {
    //@builtin(position)はclip corrdinates
    @builtin(position) clip_position: vec4<f32>,
    //@location(0) color: vec3<f32>
    @location(0) tex_coords: vec2<f32>,

    //light
    //@location(1) world_normal: vec3<f32>,
    //@location(2) world_position: vec3<f32>,
    //normal map
    @location(1) tangent_position: vec3<f32>,
    @location(2) tangent_light_position: vec3<f32>,
    @location(3) tangent_view_position: vec3<f32>,
};

//valid entry point for a vertex shader
@vertex
fn vs_main(
    model: VertexInput,
    //rust側でrender_pass.draw(0..3, 0..1);とかする
    //@builtin(vertex_index) in_vertex_index: u32,
    instance: InstanceInput
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let normal_matrix = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    //Construct the tangent matrix
    let world_normal = normalize(normal_matrix * model.normal);
    let world_tangent = normalize(normal_matrix * model.tangent);
    let world_bitangent = normalize(normal_matrix * model.bitangent);
    let tangent_matrix = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));
    
    let world_position = model_matrix * vec4<f32>(model.position, 1.0);

    //var is valiable but specify type. let can have inferred type, but cannot change value
    var out: VertexOutput;
    //out.color = model.color;
    out.clip_position = camera.view_proj * world_position;
    out.tex_coords = model.tex_coords;
    out.tangent_position = tangent_matrix * world_position.xyz;
    out.tangent_view_position = tangent_matrix * camera.view_pos.xyz;
    out.tangent_light_position = tangent_matrix * light.position;

    //out.world_normal = normal_matrix * model.normal;
    //var world_position: vec4<f32> = model_matrix * vec4<f32>(model.position, 1.0);
    //out.world_position = world_position.xyz;
    
    

    //order is important for matrix. vector is right, matrices is left
    //apply model_matrix before camerauniform.view_proj, because camera.. changes the coordinate system
    //from world space to camera space. model.. is a word space transformation.
    //out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

//Fragment shader
//@group はBindGroupLayoutとBindGroupを作るときに関係する
@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;
//normal map
@group(0) @binding(2)
var t_normal: texture_2d<f32>;
@group(0) @binding(3)
var s_normal: sampler;

//fragment shader entry point
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //light
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    //normal map
    let object_normal: vec4<f32> = textureSample(t_normal, s_normal, in.tex_coords);

    //We don't need (or want) much ambient light, so 0.1 is fine
    let ambient_strength = 0.1;
    let ambient_color = light.color * ambient_strength;
    
    //normal map
    let tangent_normal = object_normal.xyz * 2.0 - 1.0;

    let light_dir = normalize(in.tangent_light_position - in.tangent_position);
    let diffuse_strength = max(dot(tangent_normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    //specular
    let view_dir = normalize(in.tangent_view_position - in.tangent_position);
    //blinn phong. phong reflection model
    let harf_dir = normalize(view_dir + light_dir);
    //let reflect_dir = reflect(-light_dir, in.world_normal);

    //view_dir -> tangent_normalにした
    let specular_strength = pow(max(dot(tangent_normal, harf_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;
    
    return vec4<f32>(result, object_color.a);
}