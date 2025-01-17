use spirv_builder::{Capability, SpirvBuilder};
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

fn build_shader(
    path_to_create: &str,
    codegen_names: bool,
    caps: &[Capability],
) -> Result<(), Box<dyn Error>> {
    let mut builder = SpirvBuilder::new(path_to_create, "spirv-unknown-vulkan1.0");
    for &cap in caps {
        builder = builder.capability(cap);
    }
    let result = builder.build()?;
    if codegen_names {
        let out_dir = env::var_os("OUT_DIR").unwrap();
        let dest_path = Path::new(&out_dir).join("entry_points.rs");
        fs::write(&dest_path, result.codegen_entry_point_strings()).unwrap();
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    build_shader("../../shaders/sky-shader", true, &[])?;
    build_shader("../../shaders/simplest-shader", false, &[])?;
    build_shader("../../shaders/compute-shader", false, &[])?;
    build_shader(
        "../../shaders/mouse-shader",
        false,
        &[Capability::Int8, Capability::Int16, Capability::Int64],
    )?;
    Ok(())
}
