use learn_wgpu::run;


/*
 the pipeline layout, associated with the current render pipeline, contains a bind group layout at index 0 which is incompatible with the bind group layout associated with the bind group at 0
 これが出たらset_bind_groupが甘い可能性


*/
fn main() {
    
    pollster::block_on(run())
}