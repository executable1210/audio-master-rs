use crate::resampler::ResamplerQuality;

pub const MAX_AUDIO_CHANNELS: usize = 8;
pub const DEFAULT_FRAME_RATE: usize = 25;

pub const MIN_FREQ_CUT: f32 = 10.0;
pub const MAX_FREQ_CUT: f32 = 22050.0;

pub const MIN_AUDIO_MIX: f32 = 0.0;
pub const MAX_AUDIO_MIX: f32 = 1.0;
pub const DEFAULT_AUDIO_MIX: f32 = 1.0;

pub const DEFAULT_PROCESSOR_INPUT_NAME: &str = "PROCESS_INPUT";
pub const DEFAULT_PROCESSOR_OUTPUT_NAME: &str = "PROCESS_OUTPUT";

pub const MIN_SAMPLE_RATE: f32 = 1000.0;
pub const MAX_SAMPLE_RATE: f32 = 192000.0;
pub const DEFAULT_SAMPLE_RATE: f32 = 44100.0;

pub const MAX_VOLUME: f32 = 1.0;
pub const MIN_VOLUME: f32 = 0.0;
pub const DEFAULT_VOLUME: f32 = 1.0;

pub const MAX_RESAMPLE_SPEED: f32 = 4.0;
pub const MIN_RESAMPLE_SPEED: f32 = 0.2;

pub const DEFAULT_SPEED: f32 = 1.0;

pub const DEFAULT_RESAMPLER_QUALITY: ResamplerQuality = ResamplerQuality::SincMedium;

pub const MAX_RESAMPLE_RATIO: f64 = 8.0;
pub const MIN_RESAMPLE_RATIO: f64 = 0.2;