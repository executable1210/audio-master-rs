use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use fxhash::{FxHashMap, FxHashSet};
use num_traits::Float;
use rustfft::{FftPlanner, num_complex::Complex};

use crate::{
    audio_buffer::{AudioBuffer, AudioBufferInterleaved, AudioChannelLayout},
    audio_math::AudioFilter,
    backend::{AudioHost, Device, RawAudioStream},
    float_type::FloatType,
    resampler::{Resampler, ResamplerQuality},
};

/// Represents the state of an audio effect after processing or parameter operations.
/// This enum is used to indicate success or various error conditions in audio effect handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioEffectState {
    /// Processing completed successfully.
    ProcessOk,
    /// An error occurred during processing.
    ProcessError,
    /// The specified parameter does not exist.
    ParamNoEntry,
    // ParamOutOfRange, // Commented out in original; potentially for future use.
    /// The provided string parameter value is invalid, with the error message.
    ParamStrInvalid(String),
    /// The effect operation was successful.
    EffectOk,
    /// Failed to reset the effect.
    EffectResetFailed,
    /// An effect with the same name already exists.
    EffectAlreadyExists,
    /// The specified effect does not exist.
    EffectNoEntry,
}

/// Represents the value of an effect parameter, supporting various numeric and string types.
#[derive(Debug, Clone)]
pub enum EffectParamValue<'a> {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Str(&'a str),
}

/// Represents the type of an effect parameter, used for type checking and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectParamType {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F32,
    F64,
    Str,
}

/// Represents the state of an audio stream after processing.
/// This indicates whether the stream processed normally or encountered issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioStreamState {
    /// Stream processing was successful.
    Ok,
    /// An internal error occurred; input buffer is filled with zeros.
    Err,
    /// The stream is silent; input buffer is filled with zeros.
    Silence,
    /// Other error condition with a descriptive message; input buffer is filled with zeros.
    Other(String),
}

/// Volume metrics, clipping information, and FFT analysis data.
#[derive(Debug, Clone)]
pub struct AudioAnalysis {
    /// Name or identifier for this analysis.
    pub name: String,
    /// Average volume in decibels.
    pub avg_volume_db: f32,
    /// Number of clipping occurrences.
    pub clippings: usize,
    /// Ratio of clipping in the signal.
    pub clipping_ratio: f32,
    /// FFT analysis data as a vector of floats.
    pub analysis: Vec<f32>,
}

/// Alias for parameter property descriptions.
/// This is an array of tuples, each containing a parameter name and its type.
/// The const generic `N` specifies the number of parameters.
pub type ParamPropDesk<'a, const N: usize> = [(&'a str, EffectParamType); N];

/// Trait for implementing audio effects.
/// NOTE: Parameter names are validated against `ParamPropDesk`, so no additional name checks are needed.
pub trait AudioEffectTrait<T: FloatType> {
    /// Main processing function for the effect.
    /// Applies the effect to the input audio buffer.
    /// Returns the state after processing.
    fn process(&mut self, input: &mut AudioBuffer<T>) -> AudioEffectState;

    /// Retrieves the value of a parameter by name.
    /// Returns the parameter value or an error state if not found.
    fn get_param(&mut self, name: &str) -> Result<EffectParamValue, AudioEffectState>;

    /// Sets the value of a parameter by name.
    /// Returns the state after setting the parameter.
    fn set_param(&mut self, name: &str, value: EffectParamValue) -> AudioEffectState;

    /// Resets the effect to its initial state.
    /// Returns the state after reset.
    fn reset(&mut self) -> AudioEffectState;

    /// Sets the sample rate for the effect.
    /// This may be used to adjust internal calculations based on sample rate.
    fn set_sample_rate(&mut self, sample_rate: f32);
}

#[derive(Debug, Clone)]
pub struct AudioStreamContext {
    /// Timestamp of the current stream position in seconds.
    pub timestamp: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Trait for feeders that provide audio data to streams.
/// Implementors process input buffers based on the provided context.
pub trait AudioStreamFeederTrait<T: FloatType + Float> {
    /// Processes the input audio buffer using the given context.
    /// Returns the state of the stream after processing.
    fn process(
        &mut self,
        context: &AudioStreamContext,
        input: &mut AudioBuffer<T>,
    ) -> AudioStreamState;
}

/// Public API trait for audio effects.
/// NOTE: Uses raw pointers internally for reference exposure; not read/write secured for get_name() and get_params().
pub trait AudioEffectImpl<T: FloatType + Float>: Sized {
    /// Creates a new audio effect with the given name, parameter properties, and processor.
    /// The const generic `N` specifies the number of parameters.
    fn new<const N: usize>(
        name: &str,
        param_props: ParamPropDesk<N>,
        proc: Box<dyn AudioEffectTrait<T>>,
    ) -> AudioEffect<T>;

    /// Gets the name of the effect.
    fn get_name(&self) -> &str;

    /// Gets a reference to the map of all parameters and their types.
    fn get_params(&self) -> &FxHashMap<String, EffectParamType>;

    /// Sets a parameter by name.
    /// Returns the state after setting.
    fn set_param(&mut self, name: &str, value: EffectParamValue) -> AudioEffectState;

    /// Gets a parameter value by name.
    /// Returns the value or an error state.
    fn get_param(&self, name: &str) -> Result<EffectParamValue, AudioEffectState>;

    /// Enables or disables the input filter before processing.
    fn use_filter_in(&mut self, use_filter: bool);

    /// Enables or disables the output filter after processing.
    fn use_filter_out(&mut self, use_filter: bool);

    /// Checks if the input filter is enabled.
    fn is_using_filter_in(&self) -> bool;
    /// Checks if the output filter is enabled.
    fn is_using_filter_out(&self) -> bool;

    /// Sets the mix level (clamped between 0.0 and 1.0).
    fn set_mix(&mut self, new_mix: f32);

    /// Gets the current mix level.
    fn get_mix(&self) -> f32;

    /// Sets the low-cut frequency for the input filter (clamped between 10.0 and 22050.0 Hz).
    fn set_low_cut_in(&mut self, cut: f32);

    /// Gets the low-cut frequency for the input filter.
    fn get_low_cut_in(&self) -> f32;

    /// Sets the high-cut frequency for the input filter (clamped between 10.0 and 22050.0 Hz).
    fn set_high_cut_in(&mut self, cut: f32);

    /// Gets the high-cut frequency for the input filter.
    fn get_high_cut_in(&self) -> f32;

    /// Sets the low-cut frequency for the output filter (clamped between 10.0 and 22050.0 Hz).
    fn set_low_cut_out(&mut self, cut: f32);

    /// Gets the low-cut frequency for the output filter.
    fn get_low_cut_out(&self) -> f32;

    /// Sets the high-cut frequency for the output filter (clamped between 10.0 and 22050.0 Hz).
    fn set_high_cut_out(&mut self, cut: f32);

    /// Gets the high-cut frequency for the output filter.
    fn get_high_cut_out(&self) -> f32;
}

/// Public API trait for audio processors.
/// This manages a sequence of effects and global processing settings.
pub trait AudioProcessorImpl<T: FloatType + Float>: Sized {
    /// Creates a new audio processor with the given sample rate and buffer size.
    fn new(sample_rate: f32, buffer_size: usize) -> AudioProcessor<T>;

    /// Sets the global mix level (clamped between 0.0 and 1.0).
    fn set_mix(&mut self, mix: f32);

    /// Gets the global mix level.
    fn get_mix(&self) -> f32;

    /// Enables or disables the global filter.
    fn use_filter(&mut self, use_filter: bool);
    /// Checks if the global filter is enabled.
    fn is_using_filter(&self) -> bool;

    /// Sets the low-cut frequency for the global filter (clamped between 10.0 and 22050.0 Hz).
    fn set_low_cut(&mut self, low_cut: f32);

    /// Gets the low-cut frequency for the global filter.
    fn get_low_cut(&self) -> f32;

    /// Sets the high-cut frequency for the global filter (clamped between 10.0 and 22050.0 Hz).
    fn set_high_cut(&mut self, high_cut: f32);

    /// Gets the high-cut frequency for the global filter.
    fn get_high_cut(&self) -> f32;

    /// Sets the sample rate for the processor and updates all effects.
    /// Clamped between 1000.0 and 192000.0 Hz.
    fn set_sample_rate(&mut self, sample_rate: f32);

    /// Gets the current sample rate.
    fn get_sample_rate(&self) -> f32;

    /// Adds an effect to the processor.
    /// Returns `EffectAlreadyExists` if an effect with the same name is present.
    fn add_effect(&mut self, effect: AudioEffect<T>) -> AudioEffectState;

    /// Removes an effect by name.
    /// Returns the removed effect or `EffectNoEntry` if not found.
    fn remove_effect(&mut self, name: &str) -> Result<AudioEffect<T>, AudioEffectState>;

    /// Moves an effect from one position to another in the processing sequence.
    fn move_effect(&mut self, from: usize, to: usize);

    /// Gets a clone of the effect by name.
    /// Returns `EffectNoEntry` if not found.
    fn get_effect(&self, name: &str) -> Result<AudioEffect<T>, AudioEffectState>;

    /// Connects an analyzer to the processor for audio analysis.
    fn connect_analyser(&mut self, anal: Rc<RefCell<AudioAnalyser<T>>>);

    /// Disconnects and returns the connected analyzer, if any.
    fn disconnect_analyser(&mut self) -> Option<Rc<RefCell<AudioAnalyser<T>>>>;
}

/// Settings for creating an audio stream, including sample rate and channel layout.
#[derive(Debug, Clone)]
pub struct AudioStreamSettings {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Channel layout.
    pub channel_layout: AudioChannelLayout,
}

/// Error types for the audio master system.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AudioMasterError {
    /// No audio device is available.
    NoDeviceAvailable,
    /// The stream has not been initialized.
    StreamNotInitialized,
    /// Internal error with a descriptive message.
    Internal(String),
}

/// Public API trait for the audio master, which manages devices and streams.
pub trait AudioMasterImpl {
    /// Creates a new audio master with default settings.
    fn new() -> AudioMaster;
    /// Creates a new audio master with a specified buffer size.
    fn new_with_buffer_size(buffer_size: usize) -> AudioMaster;
    /// Creates a new audio master with a specified frame rate (FPS).
    fn new_with_fps(fps: usize) -> AudioMaster;

    /// Stops the main system audio stream.
    fn stop_sys_stream(&mut self) -> Result<(), AudioMasterError>;

    /// Starts the main system audio stream.
    fn start_sys_stream(&mut self) -> Result<(), AudioMasterError>;

    /// Attempts to initialize the audio stream.
    fn try_to_initialize_stream(&mut self) -> Result<(), AudioMasterError>;
    /// Gets a list of available audio devices.
    fn devices(&self) -> Vec<Device>;
    /// Gets the currently selected device, if any.
    fn get_current_device(&self) -> Option<Device>;

    /// Changes the current audio device by ID.
    fn change_device(&self, device_id: usize) -> Result<(), AudioMasterError>;

    /// Creates a new f32 audio stream with the given settings and feeder.
    fn create_stream_f32(
        &mut self,
        settings: &AudioStreamSettings,
        feeder: Box<dyn AudioStreamFeederTrait<f32>>,
    ) -> AudioStream<f32>;
    /// Creates a new f64 audio stream with the given settings and feeder.
    fn create_stream_f64(
        &mut self,
        settings: &AudioStreamSettings,
        feeder: Box<dyn AudioStreamFeederTrait<f64>>,
    ) -> AudioStream<f64>;
}

/// Public API trait for audio streams.
/// This manages stream properties like volume, speed, and filtering.
pub trait AudioStreamImpl<T: FloatType + Float> {
    /// Enables or disables normalization.
    fn use_normalization(&mut self, norm: bool);
    /// Checks if normalization is enabled.
    fn is_using_normalization(&self) -> bool;

    /// Gets the current timestamp in seconds.
    fn get_timestamp(&self) -> f64;

    /// Gets the resampler quality.
    fn get_resample_quality(&self) -> ResamplerQuality;
    /// Sets the resampler quality.
    fn set_resample_quality(&mut self, quality: ResamplerQuality);

    /// Gets the playback speed (clamped between 0.2 and 8.0).
    fn get_speed(&self) -> f32;
    /// Sets the playback speed (clamped between 0.2 and 8.0).
    fn set_speed(&mut self, speed: f32);

    /// Gets the volume level.
    fn get_volume(&self) -> f32;

    /// Sets the volume level (clamped between 0.0 and 1.0).
    fn set_volume(&self, volume: f32);

    /// Enables or disables the filter.
    fn use_filter(&mut self, filter: bool);
    /// Checks if the filter is enabled.
    fn is_using_filter(&self) -> bool;

    /// Gets the low-cut frequency.
    fn get_low_cut(&self) -> f32;
    /// Sets the low-cut frequency (clamped between 10.0 and 22050.0 Hz).
    fn set_low_cut(&mut self, low_cut: f32);

    /// Gets the high-cut frequency.
    fn get_high_cut(&self) -> f32;

    /// Sets the high-cut frequency (clamped between 10.0 and 22050.0 Hz).
    fn set_high_cut(&mut self, high_cut: f32);

    /// Gets the sample rate.
    fn get_sample_rate(&self) -> f32;

    /// Sets the sample rate (clamped between 1000.0 and 192000.0 Hz).
    fn set_sample_rate(&mut self, sample_rate: f32);

    /// Gets the channel layout.
    fn get_channel_layout(&self) -> AudioChannelLayout;
    /// Sets the channel layout.
    fn set_channel_layout(&mut self, layout: AudioChannelLayout);

    /// Gets a clone of the shared processor.
    fn get_processor(&self) -> AudioProcessor<T>;

    /// Resumes the stream if paused.
    fn resume(&mut self);
    /// Pauses the stream.
    fn pause(&mut self);
}

/// Shared pointer to an audio effect, using Arc<RwLock> for thread-safe access.
#[derive(Clone)]
pub struct AudioEffect<T: FloatType + Float> {
    /// Inner effect data, wrapped in Arc<RwLock> for shared mutable access.
    pub(super) inner: Arc<RwLock<AudioEffectInner<T>>>,
}

/// Shared pointer to the audio master.
#[derive(Clone)]
pub struct AudioMaster {
    pub(super) inner: Arc<RwLock<AudioMasterInner>>,
}

/// Shared pointer to an audio stream.
/// NOTE: This is a virtual in-runtime object, not a system-level stream.
#[derive(Clone)]
pub struct AudioStream<T: FloatType + Float> {
    pub(super) inner: Arc<RwLock<AudioStreamInner<T>>>,
}

/// Shared pointer to an audio processor.
#[derive(Clone)]
pub struct AudioProcessor<T: FloatType + Float> {
    /// Inner processor data, wrapped in Arc<RwLock> for shared mutable access.
    pub(super) inner: Arc<RwLock<AudioProcessorInner<T>>>,
}

pub trait AudioProcessorPrivateImpl<T: FloatType + Float>: Sized {
    /// Main processing function that applies effects to the input buffer.
    fn process(&mut self, input: &mut AudioBuffer<T>) -> AudioEffectState;

    /// Sets the sample rate for all effects in the processor.
    fn set_effects_sample_rate(&mut self, sample_rate: f32);

    /// Starts an analysis sequence with the given analyzer and size.
    fn anal_start_sequence(anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>, size: usize);
    /// Ends an analysis sequence with the given analyzer.
    fn anal_end_sequence(anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>);
    /// Analyzes the next node in the sequence.
    fn analyze_next_node(
        anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>,
        name: &str,
        input: &AudioBuffer<T>,
    );

    /// Sets the buffer size for the processor.
    fn set_buffer_size(&mut self, new_len: usize);
}

pub trait AudioStreamPrivateImpl<T: FloatType + Float>: Sized {
    /// Factory method to create a new stream inner with feeder, sample rate, and channel layout.
    fn factory(
        feeder: Box<dyn AudioStreamFeederTrait<T>>,
        sample_rate: u32,
        channel_layout: AudioChannelLayout,
    ) -> Self;

    /// Processes the input buffer.
    fn process(&mut self, input: &mut AudioBuffer<T>);
    /// Performs pre-processing steps.
    fn pre_process(&mut self);
    /// Performs post-processing on the input buffer.
    fn post_process(&mut self, input: &mut AudioBuffer<T>);

    /// Clears internal buffers.
    fn clear_buffers(&mut self);
    /// Sets the buffer size for the stream.
    fn set_buffer_size(&mut self, new_buff_size: usize);
}

pub trait AudioEffectPrivateImpl<T: FloatType + Float>: Sized {
    /// Processes the input buffer with the effect.
    fn process(&mut self, input: &mut AudioBuffer<T>) -> AudioEffectState;
    /// Sets the sample rate for the effect.
    fn set_sample_rate(&mut self, sample_rate: f32);
    /// Resizes internal buffers to the new length.
    fn resize(&mut self, new_len: usize);
}

pub trait AudioAnalyserPrivateImpl<T: FloatType>: Sized {
    /// Resizes internal buffers to the new length.
    fn resize(&mut self, new_len: usize);
    /// Calculates volume and clipping metrics.
    fn calc_volume_and_clippings(&mut self);
    /// Performs the analysis.
    fn analyze(&mut self);

    /// Starts an analysis sequence with the given size.
    fn start_sequence(&mut self, size: usize);
    /// Ends an analysis sequence.
    fn end_sequence(&mut self);
    /// Processes the next node in the analysis sequence.
    fn process_next_node(&mut self, name: &str, input: &AudioBuffer<T>);
}

pub trait AudioAnalysisPrivateImpl: Sized {
    fn new(len: usize) -> AudioAnalysis;
}

/// Feeder for the main audio stream.
#[derive(Clone)]
pub struct MainStreamFeeder<T> {
    /// Shared reference to the audio master inner.
    pub(super) master: Arc<RwLock<AudioMasterInner>>,
    /// Phantom data for type parameter T.
    pub(super) _phantom: PhantomData<T>,
}

/// Struct representing a parameter property.
#[derive(Debug, Clone)]
pub struct ParamProp {
    /// Name of the parameter.
    pub(super) name: String,
    /// Type of the parameter.
    pub(super) _type: EffectParamType,
}

pub trait AudioMasterPrivateImpl: Sized {
    /// Factory method to create a new master inner with host, device, buffer size, and frame rate.
    fn factory(
        host: AudioHost,
        device: Device,
        buffer_size: Option<usize>,
        frame_rate: Option<usize>,
    ) -> Arc<RwLock<Self>>;
    /// Attempts to create a default master with optional buffer size and frame rate.
    fn try_to_create_default(
        buffer_size: Option<usize>,
        frame_rate: Option<usize>,
    ) -> Option<Arc<RwLock<Self>>>;
}

pub struct AudioMasterInner {
    /// The audio host backend.
    pub(super) host: AudioHost,
    /// Currently selected device.
    pub(super) current_device: Option<Device>,
    /// List of f32 streams with their buffers.
    pub(super) streams_f32: Vec<(AudioBuffer<f32>, Arc<RwLock<AudioStreamInner<f32>>>)>,
    /// List of f64 streams with their buffers.
    pub(super) streams_f64: Vec<(AudioBuffer<f64>, Arc<RwLock<AudioStreamInner<f64>>>)>,
    /// Main raw audio stream.
    pub(super) main_stream: Option<RawAudioStream<f32>>,
}

pub struct AudioStreamInner<T: FloatType + Float> {
    /// The feeder providing audio data.
    pub(super) feeder: Box<dyn AudioStreamFeederTrait<T>>,
    /// Current sample rate.
    pub(super) sample_rate: f32,
    /// Playback speed factor.
    pub(super) speed: f32,
    /// Volume level.
    pub(super) volume: f32,
    /// Resampling factor.
    pub(super) resample_factor: f64,

    /// Channel layout.
    pub(super) channel_layout: AudioChannelLayout,
    /// Resampled interleaved buffer.
    pub(super) buffer_resampled: AudioBufferInterleaved<T>,
    /// Interleaved buffer.
    pub(super) buffer: AudioBufferInterleaved<T>,
    /// Callback buffer.
    pub(super) cb_buffer: AudioBuffer<T>,
    /// Position in the resampled buffer.
    pub(super) resampled_buffer_pos: usize,
    /// Length of the resampled buffer.
    pub(super) resampled_buffer_len: usize,

    /// Flag for using filter.
    pub(super) use_filter: bool,
    /// Flag for using normalization.
    pub(super) use_normalization: bool,
    /// Flag for using processor.
    pub(super) use_processor: bool,

    /// Associated processor.
    pub(super) processor: AudioProcessor<T>,
    /// Audio filter.
    pub(super) filter: AudioFilter<T>,

    /// Resampler instance.
    pub(super) resampler: Resampler<T>,
    /// Current timestamp.
    pub(super) timestamp: f64,
    /// Flag indicating if the stream is paused.
    pub(super) is_paused: bool,
}

pub struct AudioEffectInner<T: FloatType + Float> {
    /// Name of the effect.
    pub(super) name: String,
    /// Map of parameters and their types.
    pub(super) params: FxHashMap<String, EffectParamType>,
    /// The effect processor trait object.
    pub(super) processor: Box<dyn AudioEffectTrait<T>>,
    /// Internal buffer.
    pub(super) buffer: AudioBuffer<T>,
    /// Mix level.
    pub(super) mix: f32,

    /// Current sample rate (for getter).
    pub(super) sample_rate: f32,

    /// Input filter.
    pub(super) filter_in: AudioFilter<T>,
    /// Output filter.
    pub(super) filter_out: AudioFilter<T>,
    /// Flag for using input filter.
    pub(super) use_filter_in: bool,
    /// Flag for using output filter.
    pub(super) use_filter_out: bool,

    /// If true, mutes the input signal but allows output.
    pub(super) is_muffled: bool,
}

pub struct AudioProcessorInner<T: FloatType + Float> {
    /// Set of effect names for quick lookup.
    pub(super) effects: FxHashSet<String>,
    /// Sequence of effects to apply.
    pub(super) effects_seq: Vec<AudioEffect<T>>,
    /// Internal buffer.
    pub(super) buffer: AudioBuffer<T>,
    /// Optional analyzer.
    pub(super) analyser: Option<Rc<RefCell<AudioAnalyser<T>>>>,
    /// Global filter.
    pub(super) filter: AudioFilter<T>,
    /// Flag for using the global filter.
    pub(super) use_filter: bool,

    /// Current sample rate.
    pub(super) sample_rate: f32,
    /// Global mix level.
    pub(super) mix: f32,
}

pub struct AudioAnalyser<T: FloatType> {
    /// FFT planner for f32.
    pub(super) planner: FftPlanner<f32>,
    /// Signal data as complex numbers.
    pub(super) signal: Vec<Complex<f32>>,
    /// Mixed audio data.
    pub(super) mixed: Vec<T>,
    /// List of analyses.
    pub(super) anals: Vec<AudioAnalysis>,
    /// User callback for analysis results.
    pub(super) user_cb: Box<dyn FnMut(&[AudioAnalysis]) + 'static>,
    /// Size of the analysis sequence.
    pub(super) seq_size: usize,
    /// Current index in the sequence.
    pub(super) seq_index: usize,
    /// Flag indicating if the sequence has started.
    pub(super) seq_started: bool,
}
