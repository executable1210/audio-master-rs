use std::{
    hash::DefaultHasher,
    sync::{Arc, RwLock},
};

use cpal::{PauseStreamError, PlayStreamError, SupportedStreamConfigRange};

/// Audio host
use crate::{
    audio_buffer::{AudioBuffer, AudioChannelLayout},
    float_type::FloatType,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawAudioStreamError {
    DeviceNotAvailable,
    Internal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawAudioStreamState {
    Ok,

    /// internal error occured.
    /// Input buffer is filled with zero.
    Err,

    /// Input buffer is filled with zero.
    Silence,

    /// If an internal error occured.
    /// Input buffer is filled with zero.
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioHostError {
    NoDevice,
    NoConfig,
    UnknownSampleFormat,
    UnknownChannelCount,
    UnknownStreamConfig,
    DeviceNotAvailable,
    InvalidArgument,
    StreamIdOverflow,
    UnknownError,
    Internal(String),
}

#[derive(Debug, Clone)]
pub struct Device {
    pub name: String,
    pub sample_rate: u32,
    pub channels: AudioChannelLayout,
    pub id: usize,
}

pub trait AudioHostImpl {
    fn new() -> AudioHost;

    /// Returns default output device set by system.
    fn default_output_device(&self) -> Option<Device>;

    /// Returns a device by id.
    fn get_device_by_id(&self, id: usize) -> Result<Device, AudioHostError>;

    /// Get availabe audio devices.
    fn devices(&self) -> Vec<Device>;

    /// Create a stream with default device. Float32 type.
    /// If there's no devices returns AudioHostError::NoDevice.
    fn create_stream_default(
        &mut self,
        feeder: Box<dyn StreamFeederTrait<f32>>,
        buffer_size: usize,
    ) -> Result<RawAudioStream<f32>, AudioHostError>;

    /// Create a stream with specified device. Float64 type.
    /// If there's no devices returns AudioHostError::NoDevice.
    fn create_stream(
        &mut self,
        feeder: Box<dyn StreamFeederTrait<f32>>,
        buffer_size: usize,
        device_id: usize,
    ) -> Result<RawAudioStream<f32>, AudioHostError>;
}

pub trait RawAudioStreamImpl: Sized {
    /// Close stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_close().
    fn close(&mut self);

    /// Play stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_play().
    fn play(&mut self) -> Result<(), RawAudioStreamError>;

    /// Pause stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_pause().
    fn pause(&mut self) -> Result<(), RawAudioStreamError>;

    /// Is stream running
    fn is_running(&self) -> bool;

    /// Returns stream sample rate
    fn get_sample_rate(&self) -> u32;

    /// Returns timestamp in micro seconds
    fn get_timestamp(&self) -> usize;
}

pub trait StreamFeederTrait<T: FloatType + Send + Sync> {
    /// Main user processor.
    fn process(&mut self, input: &mut AudioBuffer<T>) -> RawAudioStreamState;

    /// Calls when an internal stream error occured, e.g - a device is turned off.
    fn emit_stream_error(&mut self, error: &str);

    /// Calls when a stream is being closed.
    fn emit_stream_close(&mut self);

    /// Calls when a stream is unpaused
    fn emit_stream_play(&mut self);

    /// Calls when a stream is paused
    fn emit_stream_pause(&mut self);
}

#[derive(Clone)]
pub struct AudioHost {
    pub(super) core: Arc<RwLock<cpal::Host>>,
}

pub struct RawAudioStreamInner<T: FloatType> {
    pub(super) core: Option<cpal::Stream>,
    pub(super) is_running: bool,
    pub(super) buffer: AudioBuffer<T>,
    pub(super) buffer_offset: usize,
    pub(super) sample_rate: u32,

    /// sample_rate / 1000
    /// Needed for timestamp fast calculation
    pub(super) sample_rate_1000: f64,
    pub(super) feeder: Box<dyn StreamFeederTrait<T>>,

    /// Count of buffer sampled passed.
    pub(super) timestamp: usize,
}

#[derive(Clone)]
pub struct RawAudioStream<T: FloatType> {
    pub(super) inner: Arc<RwLock<RawAudioStreamInner<T>>>,
}

pub trait PrivateImpl: Sized {
    /// Converts channels count to channel layout.
    fn channels_to_layout(channels: usize) -> Option<AudioChannelLayout>;

    /// Converts cpal::PlayStreamError into RawAudioStreamError.
    fn conv_cpal_play_stream_error(error: PlayStreamError) -> RawAudioStreamError;

    /// Converts cpal::PauseStreamError into RawAudioStreamError.
    fn conv_cpal_pause_stream_error(error: PauseStreamError) -> RawAudioStreamError;

    /// Converts cpal::Device into Device.
    fn cpal_device_to_device(device: cpal::Device) -> Option<Device>;

    /// Generates id from cpol::Device.
    /// Device's content used as ssid.
    fn generate_id_from_cpal_device(device: &cpal::Device) -> i64;
    fn hash_config_range(config: &SupportedStreamConfigRange, hasher: &mut DefaultHasher);
}

pub trait AudioHostPrivateImpl: Sized {
    /// Try to find cpal::Device by id
    fn get_cpal_device_by_id(&self, id: usize) -> Result<cpal::Device, AudioHostError>;

    /// Try to create stream with specific device
    fn create_stream_with_device<T: FloatType + Send + Sync + 'static>(
        &mut self,
        device: &cpal::Device,
        buffer_size: usize,
        feeder: Box<dyn StreamFeederTrait<T>>,
    ) -> Result<Arc<RwLock<RawAudioStreamInner<T>>>, AudioHostError>;
}

pub trait RawAudioStreamPrivateImpl<T: FloatType + Send + Sync>: Sized {
    /// Main process function for buffer size and channel layout alignments
    fn user_cpal_process(&mut self, buffer: &mut [f32], channels: AudioChannelLayout);
}

pub struct Private {}
