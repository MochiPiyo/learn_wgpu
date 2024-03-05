use anyhow::*;
use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use std::env;

//ビルド時実行


fn main() -> Result<()> {
    //this tells cargo to rerun this script if something in /res/ changes.
    println!("cargo: rerun-if-canged=res/*");

    //OUT_DIR is an environment variable that cargo uses to specify where out application will be build
    let out_dir = env::var("OUT_DIR")?;
    let mut copy_options = CopyOptions::new();
    copy_options.overwrite = true;
    let mut paths_to_copy = vec!["res/"];
    copy_items(&paths_to_copy, out_dir, &copy_options)?;

    Ok(())
}
