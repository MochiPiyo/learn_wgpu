use anyhow::*;
use image::{DynamicImage, GenericImageView};
use std::io::Read;

use crate::InstanceObject;

#[derive(PartialEq)]
pub enum ImageType {
    Jpg,
    Png,
}
pub struct Image<'a> {
    pub path: &'a str,
    pub image_type: ImageType,
}

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
}
impl Texture {
    pub fn new_from_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        file_name: &str,
        is_normal_map: bool,
    ) -> Result<Self> {
        let dynamic_image = image::load_from_memory(bytes)?;

        let img_source = Image {
            path: "cow.jpg",
            image_type: ImageType::Jpg,
        };
        return Self::dynamic_image_to_texture(device, queue, img_source, &dynamic_image, is_normal_map)
    }

    pub fn new_from_path(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img_source: Image,
        is_normal_map: bool,
    ) -> Result<Self> {
        

        //include_bytes!だとコンパイル時に決定せよと言われるので
        let mut file = std::fs::File::open(img_source.path).expect("failed to file open");
        //file into bytes
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).expect("failed to read");

        //&だとimgのライフタイムはこの関数内で尽きるがBoxにすると取りまわせる。有能だ
        let dynamic_image = image::load_from_memory(&bytes).expect("failed to load from memory");
        
        return Self::dynamic_image_to_texture(device, queue, img_source, &dynamic_image, is_normal_map)

        /* これはだめらしい。最初のmatchで消費される。
        {
            let rgba = match img_source.image_type {
                ImageType::Png => dynamic_image.as_rgba8().unwrap(),
                ImageType::Jpg => &(dynamic_image.to_rgba8()),
            };
        }
        //これもだめ。dynamic..がmatchの対象ではないからではない。
        let rgba = match (dynamic_image, img_source.image_type) {
            (_, ImageType::Png) => dynamic_image.as_rgba8().unwrap(),
            (_, ImageType::Jpg) => &dynamic_image.to_rgba8(),
            (_, _) => panic!(),
        };
        refは効果なし
        let rgba = match (dynamic_image, img_source.image_type) {
            (ref png, ImageType::Png) => png.as_rgba8().unwrap(),
            (ref jpg, ImageType::Jpg) => &(jpg.to_rgba8()),
            (_, _) => panic!(),
        };
        */
    }

    

    fn dynamic_image_to_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img_source: Image,
        dynamic_image: &DynamicImage,
        is_normal_map: bool,
    ) -> Result<Texture> {
        let label = format!("image path='{:?}'", img_source.path);
        match img_source.image_type {
            ImageType::Png => {
                //to_rgba8() is for jpeg, as_rgba8() is for png
                let rgba = dynamic_image.as_rgba8().unwrap();
                let dimensions = dynamic_image.dimensions();
                return Self::rgba_to_texture(device, queue, rgba, dimensions, Some(&label), is_normal_map);
            }
            ImageType::Jpg => {
                //to_rgba8() is for jpeg, as_rgba8() is for png
                let rgba = &dynamic_image.to_rgba8();
                let dimensions = dynamic_image.dimensions();
                return Self::rgba_to_texture(device, queue, rgba, dimensions, Some(&label), is_normal_map);
            }
        };
    }

    //後処理の共通化
    fn rgba_to_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rgba: &image::RgbaImage,
        dimensions: (u32, u32),
        label: Option<&str>,
        is_normal_map: bool,
    ) -> Result<Self> {
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if is_normal_map {
                wgpu::TextureFormat::Rgba8Unorm
            }else {
                wgpu::TextureFormat::Rgba8UnormSrgb
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * dimensions.0),
                rows_per_image: std::num::NonZeroU32::new(dimensions.1),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
        })
    }

    //to create depth stage of the render_pipeline, and depth texture itself.
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub fn create_depth_texture(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, label: &str) -> Self {
        //depth texture needs to be the same size as screen
        let size = wgpu::Extent3d {
            //use config, same size as surface
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        };
        let desc = wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::DEPTH_FORMAT,
            //render_attachment to rendering to this texture
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        };
        let texture = device.create_texture(&desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(
            &wgpu::SamplerDescriptor {
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                //need for render depth texture
                compare: Some(wgpu::CompareFunction::LessEqual),
                lod_min_clamp: -100.0,
                lod_max_clamp: 100.0,
                ..Default::default()
            }
        );

        Self {
            texture,
            view,
            sampler,
        }
    }
}
