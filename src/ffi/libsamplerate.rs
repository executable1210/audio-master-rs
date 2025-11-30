// Automatically generated Rust bindings for libsamplerate
use std::os::raw::{c_char, c_double, c_float, c_int, c_long, c_short, c_void};

// Opaque type for SRC_STATE
#[repr(C)]
pub struct SRC_STATE {
    _private: [u8; 0],
}

// SRC_DATA struct for passing data to src_simple and src_process
#[repr(C)]
pub struct SRC_DATA {
    pub data_in: *const c_float,
    pub data_out: *mut c_float,
    pub input_frames: c_long,
    pub output_frames: c_long,
    pub input_frames_used: c_long,
    pub output_frames_gen: c_long,
    pub end_of_input: c_int,
    pub src_ratio: c_double,
}

// Callback type for src_callback_new and src_callback_read
pub type src_callback_t =
    Option<unsafe extern "C" fn(cb_data: *mut c_void, data: *mut *mut c_float) -> c_long>;

// Converter types as enum
#[repr(C)]
pub enum SRC_CONVERTER_TYPE {
    SRC_SINC_BEST_QUALITY = 0,
    SRC_SINC_MEDIUM_QUALITY = 1,
    SRC_SINC_FASTEST = 2,
    SRC_ZERO_ORDER_HOLD = 3,
    SRC_LINEAR = 4,
}

unsafe extern "C" {
    // Initialize a new converter
    pub fn src_new(converter_type: c_int, channels: c_int, error: *mut c_int) -> *mut SRC_STATE;

    // Clone an existing converter
    pub fn src_clone(orig: *mut SRC_STATE, error: *mut c_int) -> *mut SRC_STATE;

    // Initialize callback-based converter
    pub fn src_callback_new(
        func: src_callback_t,
        converter_type: c_int,
        channels: c_int,
        error: *mut c_int,
        cb_data: *mut c_void,
    ) -> *mut SRC_STATE;

    // Clean up converter
    pub fn src_delete(state: *mut SRC_STATE) -> *mut SRC_STATE;

    // Process audio data
    pub fn src_process(state: *mut SRC_STATE, data: *mut SRC_DATA) -> c_int;

    // Callback-based read
    pub fn src_callback_read(
        state: *mut SRC_STATE,
        src_ratio: c_double,
        frames: c_long,
        data: *mut c_float,
    ) -> c_long;

    // Simple interface for single conversion
    pub fn src_simple(data: *mut SRC_DATA, converter_type: c_int, channels: c_int) -> c_int;

    // Get converter name and description
    pub fn src_get_name(converter_type: c_int) -> *const c_char;
    pub fn src_get_description(converter_type: c_int) -> *const c_char;
    pub fn src_get_version() -> *const c_char;

    // Set conversion ratio
    pub fn src_set_ratio(state: *mut SRC_STATE, new_ratio: c_double) -> c_int;

    // Get channel count
    pub fn src_get_channels(state: *mut SRC_STATE) -> c_int;

    // Reset converter state
    pub fn src_reset(state: *mut SRC_STATE) -> c_int;

    // Validate conversion ratio
    pub fn src_is_valid_ratio(ratio: c_double) -> c_int;

    // Get error code
    pub fn src_error(state: *mut SRC_STATE) -> c_int;

    // Convert error code to string
    pub fn src_strerror(error: c_int) -> *const c_char;

    // Helper functions for data conversion
    pub fn src_short_to_float_array(input: *const c_short, output: *mut c_float, len: c_int);
    pub fn src_float_to_short_array(input: *const c_float, output: *mut c_short, len: c_int);
    pub fn src_int_to_float_array(input: *const c_int, output: *mut c_float, len: c_int);
    pub fn src_float_to_int_array(input: *const c_float, output: *mut c_int, len: c_int);
}
