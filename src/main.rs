use learn_wgpu::run;
use pollster::block_on;

fn main() {
    pollster::block_on(run())
}