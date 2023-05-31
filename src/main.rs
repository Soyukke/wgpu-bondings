use rand::Rng;
use rand::distributions::{Distribution, Standard};

use std::{thread, time};
use bytemuck::{Pod, Zeroable};
use std::{borrow::Cow, mem};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window, dpi::LogicalSize,
};
use wgpu::util::DeviceExt;

#[derive(Debug, Clone)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance_from(&self, o: &Point) -> f64 {
        f64::sqrt((self.x - o.x).powi(2) + (self.y - o.y).powi(2))
    }

    fn mul(&self, other: &Point) -> Point {
        Point {x: self.x * other.x, y: self.y * other.y}
    }

}

impl Distribution<Point> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Point {
        let (rand_x, rand_y): (f64, f64) = rng.gen();
        Point {
            x: rand_x,
            y: rand_y,
        }
    }
}


#[derive(Debug)]
struct Obj {
    p: Point,
    v: Point,
}

impl Obj {
    fn new(s: (f64, f64)) -> Self {
        let mut rng = rand::thread_rng();
        let p2 = Point {x: s.0, y: s.1};
        let p0: Point = rng.gen();
        let p1: Point = rng.gen();
        Self {
            p: p0.mul(&p2),
            v: p1.mul(&p2),
        }
    }

    fn step(&mut self, dt: f64) {
        self.p.x += self.v.x * dt;
        self.p.y += self.v.y * dt;
    }

    fn periodic(&mut self, lx: f64, ly: f64) {
        if lx < self.p.x {
            self.p.x -= lx;
        }
        if ly < self.p.y {
            self.p.y -= ly;
        }

        if self.p.x < 0. {
            self.p.x += lx;
        }
        if self.p.y < 0. {
            self.p.y += ly;
        }
    }

    /// 最小の距離を計算する
    fn distance_from_parallel(&self, other: &Point, area: (f64, f64)) -> (Point, f64) {
        let ps = self.parallel_points(area);
        let ds: Vec<f64> = ps.iter().map(|p| p.distance_from(&other)).collect();
        let min = ds.clone().into_iter().fold(f64::NAN, f64::min);
        let idx = ds.iter().position(|&x| x==min).unwrap();
        (ps[idx].clone(), min)
    }

    fn parallel_points(&self, area: (f64, f64)) -> Vec<Point> {
        let mut ps = vec![];
        for l in -1..=1 {
            for m in -1..=1 {
                ps.push(Point {x: self.p.x + f64::from(l)*area.0, y: self.p.y + f64::from(m)*area.1});
            }
        }
        return ps;
    }

}

struct Bond {
    p1: Point,
    p2: Point,
}

struct Bonds {
    objs      : Vec<Obj>,
    bonds     : Vec<(usize, usize)>,
    bondposs  : Vec<Bond>,
    dt        : f64,
    area      : (f64, f64),
    threshold : f64,
}

impl Bonds {
    fn new(n: usize, area: (f64, f64)) -> Self {
        let mut objs = vec![];
        for _ in 0..n {
            objs.push(Obj::new(area));
        }
        Self { objs, bonds: vec![], bondposs: vec![], dt: DELTA_TIME, area, threshold: BOND_LENGTH }
    }

    fn step(&mut self) {
        for o in self.objs.iter_mut() {
            o.step(self.dt);
            o.periodic(self.area.0, self.area.1)
        }
    }

    fn update_bonds(&mut self) {
        self.bonds = vec![];
        self.bondposs = vec![];
        for i in 0..self.objs.len() {
            for j in i+1..self.objs.len() {
                let (p, d) = self.objs[i].distance_from_parallel(&self.objs[j].p, self.area);
                if d < self.threshold {
                    self.bonds.push((i, j));
                    self.bondposs.push(Bond {p1: p, p2: self.objs[j].p.clone()});
                }
            }
        }
    }
}



const WINDOW_WIDTH : f32  = 200.0;
const TEX_WIDTH    : f32  = 10.0;
const N_MAX        : usize = 100;
const N_BOND_MAX   : usize = 1000;
const BOND_LENGTH  : f64 = 30.;
const DELTA_TIME   : f64 = 0.002;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp: [[f32; 4]; 4], // matrix
    size: [f32; 2], // textureのサイズ
    pad: [f32; 2], // structのメモリ配列のあまりかな
}

/**
 * 座標(x, y): [f32, 2]
 * 
 */
#[repr(C, align(256))]
#[derive(Clone, Copy, Zeroable)]
struct Locals {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
    _pad: u32,
}

/**
 * 座標(x, y): [f32, 2]
 * 
 */
#[repr(C, align(256))]
#[derive(Clone, Copy, Zeroable)]
struct Line {
    p: [f32; 4],
    color: u32,
    _pad: u32,
}

/// 指定の画像をテクスチャとして読み込む
fn get_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
        let texture = {
            let img_data = include_bytes!("testtexture.png");
            let dyimg = image::load_from_memory(img_data).unwrap();
            let dyimg2 = dyimg.into_rgba8();

            let size = wgpu::Extent3d {
                width: dyimg2.width(),
                height: dyimg2.height(),
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            });
            queue.write_texture(
                texture.as_image_copy(),
                &dyimg2.into_raw(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZeroU32::new(size.width * 4),
                    rows_per_image: None,
                },
                size,
            );
            texture
        };
        texture
}

async fn run(event_loop: EventLoop<()>, window: Window, mut bonds: Bonds) {

    let size = window.inner_size();
    println!("size: {:?}", size);
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                // Make sure we use the texture resolution limits from the adapter, so we can support images the size of the swapchain.
                limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });
    let shader_line = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader_line.wgsl"))),
    });


    let global_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(mem::size_of::<Globals>() as _),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                ],
                label: None,
        });



    let local_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
                },
                count: None,
            }],
            label: None,
        });

    let local_line_bind_group_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(mem::size_of::<Line>() as _),
                },
                count: None,
            }],
            label: None,
        });



    let view = get_texture(&device, &queue).create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let swapchain_capabilities = surface.get_supported_formats(&adapter);
    let swapchain_format = swapchain_capabilities[0];
    println!("swapchain_format: {:?}", swapchain_format);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        // alpha_mode: swapchain_capabilities.alpha_modes[0],
        alpha_mode: wgpu::CompositeAlphaMode::PostMultiplied,
    };

    let globals = Globals {
        mvp: glam::Mat4::orthographic_rh(
                 0.0,
                 WINDOW_WIDTH,
                 0.0,
                 WINDOW_WIDTH,
                 -1.0,
                 1.0,
                 )
            .to_cols_array_2d(),
            size: [TEX_WIDTH; 2],
            pad: [0.0; 2],
    };

    let global_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("global"),
        contents: bytemuck::bytes_of(&globals),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });


    let uniform_alignment =
        device.limits().min_uniform_buffer_offset_alignment as wgpu::BufferAddress;
    let local_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("local"),
        size: (N_MAX as wgpu::BufferAddress) * uniform_alignment,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });
    let local_line_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("line"),
        size: (N_BOND_MAX as wgpu::BufferAddress) * uniform_alignment,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });



    let global_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &global_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: global_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: None,
    });


    let local_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &local_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &local_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
            }),
        }],
        label: None,
    });

    let local_line_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &local_line_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &local_line_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(mem::size_of::<Locals>() as _),
            }),
        }],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline_layout2 = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&global_bind_group_layout, &local_line_bind_group_layout],
        push_constant_ranges: &[],
    });


    // テクスチャの表示
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        // 矩形の4点で描画する
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            strip_index_format: Some(wgpu::IndexFormat::Uint16),
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    // bondの表示
    let render_pipeline_bond = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout2),
        vertex: wgpu::VertexState {
            module: &shader_line,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_line,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        // 矩形の4点で描画する
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..wgpu::PrimitiveState::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {

        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Window sizeにする
                surface.configure(&device, &config);
                window.request_redraw();
            }
            // update
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {

                    // localsに書き込む
                    let mut rng = rand::thread_rng();
                    let color = rng.gen::<u32>();
                    bonds.step();
                    bonds.update_bonds();
                    let mut rects: Vec<Locals> = vec![];
                    let mut lines: Vec<Line> = vec![];
                    for b in bonds.bondposs.iter() {
                        if lines.len() == N_BOND_MAX {
                            break;
                        }
                        let line = [b.p1.x as f32, b.p1.y as f32, b.p2.x as f32, b.p2.y as f32];
                        let power = 255. * (1.0 - (f64::sqrt((b.p2.x-b.p1.x).powi(2) + (b.p2.y-b.p1.y).powi(2)) / BOND_LENGTH));
                        lines.push(Line {p: line, color: power as u32, _pad: 0});
                    }
                    for obj in &bonds.objs {
                        rects.push(Locals {
                            position: [obj.p.x as f32, obj.p.y as f32],
                            velocity: [0.0, 0.0],
                            color,
                            _pad: 0,
                        });
                    }
                    let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment;
                    queue.write_buffer(&local_buffer, 0, unsafe {
                        std::slice::from_raw_parts(
                            rects.as_ptr() as *const u8,
                            rects.len() * uniform_alignment as usize,
                            )
                    });

                    queue.write_buffer(&local_line_buffer, 0, unsafe {
                        std::slice::from_raw_parts(
                            lines.as_ptr() as *const u8,
                            lines.len() * uniform_alignment as usize,
                            )
                    });

                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                //load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &global_group, &[]);
                    for i in 0..rects.len() {
                        let offset =
                            (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
                        rpass.set_bind_group(1, &local_group, &[offset]);
                        rpass.draw(0..4, 0..1);
                    }

                    rpass.set_pipeline(&render_pipeline_bond);
                    rpass.set_bind_group(0, &global_group, &[]);
                    for i in 0..lines.len() {
                        if N_BOND_MAX < i {
                            break;
                        }
                        let offset =
                            (i as wgpu::DynamicOffset) * (uniform_alignment as wgpu::DynamicOffset);
                        rpass.set_bind_group(1, &local_line_group, &[offset]);
                        rpass.draw(0..2, 0..1);
                    }

                }

                queue.submit(Some(encoder.finish()));
                frame.present();

                // 無限ループにする
                let ten_millis = time::Duration::from_millis(10);
                thread::sleep(ten_millis);
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {

    let area = (WINDOW_WIDTH as f64, WINDOW_WIDTH as f64);
    let bonds = Bonds::new(N_MAX, area);

    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new()
                .with_decorations(true)
                .with_always_on_top(false)
                .with_inner_size(LogicalSize::new(WINDOW_WIDTH, WINDOW_WIDTH))
                .with_transparent(false);
    builder = builder.with_title("Bondings");
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        // Temporarily avoid srgb formats for the swapchain on the web
        pollster::block_on(run(event_loop, window, bonds));
    }
}


