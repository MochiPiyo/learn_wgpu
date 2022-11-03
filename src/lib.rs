use wgpu::{Surface, util::DeviceExt, RenderPass};
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop, self},
    window::{WindowBuilder, Window}, dpi::PhysicalPosition,
};

//wasm
#[cfg(target_arch="wasm32")]
use wasm_bindgen::prelude::*;

use bytemuck;

mod texture;

//bytemuck::PodはこのstructがPlain Old Dataであることを示し&[u8]として解釈できるようにする
//bytemuck::Zeroableはstd::mem::zeroed()を使用できることを示す
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    //x,y,z
    position: [f32; 3],
    //red,green,blue
    //color: [f32;3],
    tex_coords: [f32; 2],
}
//if the struct includes types which don't implement, impl insted of [derive()]
//unsafe impl bytemuck::Pod for Vertex {}
//unsafe impl bytemuck::Zeroable for Vertex {}

//used in create_render_pipeline
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            //how wide a vertex is, in this case it will probably be 24bytes
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            //buffer array type, which is array[vertex] or array[instance]
            step_mode: wgpu::VertexStepMode::Vertex,
            //describe the individual parts of the vertex
            //in this case, Vertex.positon and Vertex.color became array of 48bit elements like this
            //.. | buffer element | ..
            //.. |position| color | ..
            //.. | 24bit  | 24bit | ..
            attributes: &[
                //attribution of Vertex.position
                wgpu::VertexAttribute {
                    //size_of previous data, first one is usually 0.
                    offset: 0,
                    //location of this attibute. 
                    //Vertex.position correspond @location(0) x: vec3<f32> in .wgsl
                    shader_location: 0,
                    //Float32x3 correspond to vec3<f32>
                    format: wgpu::VertexFormat::Float32x3,
                },
                //Vertex.color
                wgpu::VertexAttribute {
                    //offset size fo previous attribute
                    offset: std::mem::size_of::<[f32;3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                }
            ]
            /*or write attributes with vertex_attr_array! macro
            arributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
            //but result of this macro is temporary value, so change the lifetime to 'static or const
            (before fn desc() but in impl)
            const ATTR: [wgpu::VertexAttribute; 2] = &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
            (replace attributes: &[wgpu::VertexAttribute {}, ..] in wgpu::VertexBufferLayout{})
            attributes: &self::ATTR
            */
        }
    }
}

//counter-clockwise order is front_face of the triangle (because specified in render_pipeline)
//non front_face triangle will be culled
/*
const VERTICES: &[Vertex] = &[
    //top
    Vertex { position: [0.0, 0.5, 0.0], color: [1.0, 0.0, 0.0] },
    //bottom left
    Vertex { position: [-0.5, -0.5, 0.0], color: [0.0, 1.0, 0.0] },
    //bottom right
    Vertex { position: [0.5, -0.5, 0.0], color: [0.0, 0.0, 1.0] },
];*/
/*
const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, 0.49240386, 0.0], color: [0.5, 0.0, 0.5] }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], color: [0.5, 0.0, 0.5] }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], color: [0.5, 0.0, 0.5] }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], color: [0.5, 0.0, 0.5] }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], color: [0.5, 0.0, 0.5] }, // E
];*/
// show image
const VERTICES: &[Vertex] = &[
    // Changed
    Vertex { position: [-0.0868241, 0.49240386, 0.0], tex_coords: [0.4131759, 0.00759614], }, // A
    Vertex { position: [-0.49513406, 0.06958647, 0.0], tex_coords: [0.0048659444, 0.43041354], }, // B
    Vertex { position: [-0.21918549, -0.44939706, 0.0], tex_coords: [0.28081453, 0.949397], }, // C
    Vertex { position: [0.35966998, -0.3473291, 0.0], tex_coords: [0.85967, 0.84732914], }, // D
    Vertex { position: [0.44147372, 0.2347359, 0.0], tex_coords: [0.9414737, 0.2652641], }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}
impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

use cgmath;
struct Camera {
    eye: cgmath::Point3<f32>,
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
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        //camera
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        //presentate depth, without this, far and close object are same size
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        //
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}



struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    //shader.wgsl
    render_pipeline: wgpu::RenderPipeline,
    //buffer
    vertex_buffer: wgpu::Buffer,
    //num of vertices to buffer
    //num_vertices: u32,

    //indexing Vertex
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    //bind group to show image
    diffuse_bind_group: wgpu::BindGroup,

    //Texture struct
    diffuse_texture: texture::Texture,

    
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}
impl State {
    //creating some of the wgpu types requires async code
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        //the instance is a handle to our GPU
        //backends::all => Vulkan + Metal + DX12 + Bouser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        //surfaceに直接書き込む
        let surface = unsafe {
            instance.create_surface(window)
        };

        //GPUとの接続。どのようなGPUがあるか
        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ).await.unwrap();

        //this allows specify what extra features we want
        let devive_descripter = &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                //webGL doesn't support all of wgpu's features,
                //so if we're building for the web we'lll have to disable some.
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                }else {
                    wgpu::Limits::default()
                },
                label: None,
            };
        let (device, queue) = adapter.request_device(devive_descripter, None).await.unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_supported_formats(&adapter)[0],
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
        };
        surface.configure(&device, &config);

        //show image
        //"image" depend on rayon.rs to sppd up, so slower in web. use builtin decoders with wasm-bindgen
        let diffuse_bytes = include_bytes!("machine.png");
        //textures.rs を使う
        let diffuse_texture = texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "machine.png").expect("err in from_bytes() in testure.rs");
        
        /*これはtexture.rsに行った
        let diffuse_image = image::load_from_memory(diffuse_bytes).expect("failed to load image");
        //ここがto_rgb8になっていて不足したバッファと入れるべきデータのサイズ割り算したら3:4が出てきて、
        //確かにこれはここにaがないせいで４つあるはずの色が３つになっていたんだなと
        let diffuse_rgba = diffuse_image.to_rgba8();

        use image::GenericImageView;
        //size of image dimension
        let dimensions = diffuse_image.dimensions();

        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let diffuse_texture = device.create_texture(
            &wgpu::TextureDescriptor {
                //All textures are stored as 3D, we represent out 2D texture by setting depth to 1
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                //Most images are stored using sRGB so we need to reflect that here
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                //TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
                //COPY_DST means that we want to copy data to this texture
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some("diffuse_texture"),

            }
        );
        queue.write_texture(
            //Tells wgpu where to copy the pixel data
            wgpu::ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            //The actual pixel data
            &diffuse_rgba,
            //The layout of the texture
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );
        //configure the texture view
        let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            //ClampToEdge means that outside the textre will return the color of the nearest pixel, on the edges.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            //sample footprint is smaller or larger than drawing area, mag and min usually work when far from or close to camera.
            //Linear: linear interpolation between color around this pixel
            //Nearest: same as nearest pixel. good for when it's far from camera
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            //how to blend mag/min
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });*/
        //describes a set ot resources and how they can be accessed by a shader
        let texture_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    //binding(0)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        //only visible to the fragment shader
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    //binding(0)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        //This should match the filterable field of the correspongding Texture entry above
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            }
        );
        let diffuse_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                    },
                ],
                label: Some("diffuse_bind_group"),
            }
        );

        //camera
        let camera = Camera {
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
        };

        //my struct CameraUniform
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let camera_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }
                ],
                label: Some("camera_bind_group_layout"),
            }
        );
        let camera_bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: &camera_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: camera_buffer.as_entire_binding(),
                    }
                ],
                label: Some("camera_bind_group"),
            }
        );
        
        


        //to use shader.wgsl
        //or let shader = device.create_shader_module(include_wgsl!("shader.wgsl"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        /*
        wgpu::BindGroupDescriptor { entries: &[index] } -> binding(index)
        wgpu::PipelineLayoutDescriptor { bind_group_layouts: &[index]} -> group(index)

        groupの中にbindがある。bind_groupはgroup[bind]ってことか
        pub bind_group_layouts: &[&BindGroupLayout]
        
        group(group_index) binding(bind_index)
        */
        let render_pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    //this index is bind @group(index) in shader.wgsl
                    //@group(0)
                    &texture_bind_group_layout,
                    //@group(1)
                    &camera_bind_group_layout,
                ],
                push_constant_ranges: &[],
            }
        );
        

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                //function which @vertex in .wgsl
                entry_point: "vs_main",
                //vertex to pass shader
                buffers: &[
                    Vertex::desc(),
                ],
            },
            //fragment is technically optional, so wrap it Some()
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                //function which @fragment in .wgsl
                entry_point: "fs_main",
                //set up color types for output
                targets: &[
                    Some(wgpu::ColorTargetState {
                        //we only need for surface's format for surcace created before
                        format: config.format,
                        //just replace old pixel data
                        blend: Some(wgpu::BlendState::REPLACE),
                        //all color: red, blue,green, and alpha
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                ],
            }),
            //hou to convert vertices into triangles
            primitive: wgpu::PrimitiveState {
                //every three vertices will correspond to one triangle
                topology: wgpu:: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                //how to determin which side triangle's face is
                front_face: wgpu::FrontFace::Ccw,
                //cull (delete) triangles which not facing forward
                cull_mode: Some(wgpu::Face::Back),
                
                //setting ths to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                //requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                //requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            //
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                //how many samples the pipeline will use
                count: 1,
                //which samples should be active. in this case, all.
                mask: !0,
                //not use anti-aliasing
                alpha_to_coverage_enabled: false,
            },
            multiview: None,

        });


        //use buffer
        let vertex_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        //create buffer index. this works as pointer for vertex to decrease memory use
        let index_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Index Buffer"),
                contents: bytemuck::cast_slice(INDICES),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        //buffer size for this 
        let num_indices = INDICES.len() as u32;

        //ようやくreturn
        return Self {
            surface,
            device,
            queue,
            config,
            size,

            //to use shader.wgsl
            render_pipeline,
            vertex_buffer,

            index_buffer,
            num_indices,

            diffuse_bind_group,
            diffuse_texture,

            camera,
            camera_buffer,
            camera_uniform,
            camera_bind_group,
        };
    }
    
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        return false;
    }
    fn update(&mut self) {
        //none
    }
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        //コマンドバッファーを作る
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") }
        );

        //{}はencoderの所有権
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        //backgroud color???
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,       
                    },
                })],
                depth_stencil_attachment: None,
            });
            
            //to use render pipeline
            //set the render_pipeline which created at fn new()
            render_pass.set_pipeline(&self.render_pipeline);

            //show image, 
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);

            //camera
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);

            //0 is buffer slot
            //.slice(..) is buffer area to use
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));

            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            //we tell wgpu to draw something with 3 vertices, and 1 instance. 
            //this is whre @builtin(vertex_index) comes from
            //render_pass.draw(0..self.num_vertices, 0..1);

            
            

            //draw with index buffer
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        

        //submit will accept anything that implements INtoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

//wasmが開始した時にこれを実行させる
#[cfg_attr(target_arch="wasm32", wasm_bindgen(start))]
pub async fn run() {

    //window setup
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        }else {
            env_logger::init();
        }
    }
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    #[cfg(target_arch="wasm32")]
    {
        //winit prevents sizing with CSS, so we have to set the size manually when on web
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dist = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
        }).expect("Couldn't append canvas to document body");
    }

    let mut state = State::new(&window).await;

    //Event loop
    event_loop.run(move |event, _, contorl_flow| 
        match event {
            Event::WindowEvent { 
                ref event,
                window_id,
            } if window_id == window.id() => if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested | WindowEvent::KeyboardInput {
                        input: KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                        ..
                    } => *contorl_flow = ControlFlow::Exit,
    
                    //大きさが変わったとき
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, ..} => {
                        //new_inner_size is &&mut so we have to dereference it twice
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                };
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    //reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    //the sytem is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *contorl_flow = ControlFlow::Exit,
                    //All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            Event::MainEventsCleared => {
                //RedrawRequested will only trigger once, unless we manually request if.
                window.request_redraw();
            }
            
            _ => {}
        }
    );

    
}
