use std::{env, fs, path::Path, path::PathBuf, process::Command};

fn build_sample_rate() {
    let vendor_path = Path::new("vendors/libsamplerate");

    // --- Clone repo if missing ---
    if !vendor_path.exists() {
        fs::create_dir_all("vendors").expect("Failed to create vendors directory");

        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/libsndfile/libsamplerate",
                "vendors/libsamplerate",
            ])
            .status()
            .expect("Failed to run git");

        if !status.success() {
            panic!("git clone failed");
        }
    }

    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let mut config = cmake::Config::new("vendors/libsamplerate");

    if profile == "release" {
        config.define("BUILD_SHARED_LIBS", "OFF");
        config.profile("Release");

        let dst = config.build();
        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=static=samplerate");
    } else {
        config.define("BUILD_SHARED_LIBS", "ON");
        config.profile("Debug");

        
        let dst = config.build();

        println!("cargo:rustc-link-search=native={}/lib", dst.display());
        println!("cargo:rustc-link-lib=samplerate");

        // --- Copy DLL ---
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_dir = out_dir.ancestors().nth(3).unwrap().to_path_buf();

        let dll_src = dst.join("bin").join("samplerate.dll");
        let dll_dst = target_dir.join("samplerate.dll");

        if dll_src.exists() {
            fs::copy(&dll_src, &dll_dst).expect("Failed to copy samplerate.dll");
        } else {
            panic!("Could not find DLL at {}", dll_src.display());
        }
    }

    println!("cargo:rerun-if-changed=vendors/libsamplerate/CMakeLists.txt");
}

fn main() {
    #[cfg(feature = "libsamplerate")]
    build_sample_rate();
}
