use biquad::DirectForm1;
use num_traits::Float;

use crate::{
    audio_buffer::{AudioBuffer, AudioChannelLayout},
    float_type::FloatType,
};

pub trait AudioMathImpl<T: FloatType + Clone> {
    // fn mix_signals_inplace_normalized(a: &mut [T], b: &[T], gain: f32);
    fn mix_signals_inplace(a: &mut [T], b: &[T], gain: f32);
    fn mix_signals(a: &mut [T], b: &[T]);

    fn normalize_rms(buffer: &mut [T], rms_factor: T);
    fn normalize(buffer: &mut [T], factor: T);

    /// Scale audio buffer by dB value
    fn scale_volume_db(audio: &mut [T], db: T);

    /// Scale audio buffer by dB with clipping protection
    fn scale_volume_db_clipped(audio: &mut [T], db: T);

    /// Normalize audio to specific maximum dB level
    fn normalize_to_max_db(audio: &mut [T], target_max_db: T);

    /// Normalize with headroom (e.g., -3dB headroom)
    fn normalize_with_headroom(audio: &mut [T], target_max_db: T, headroom_db: T);
}

pub trait AudioFilterImpl<T: FloatType + Float>: Sized {
    /// Creates a new bandpass filter with specified low and high cutoff frequencies
    fn new(
        low_cut: f32,
        high_cut: f32,
        sample_rate: f32,
        channel_layout: AudioChannelLayout,
    ) -> AudioFilter<T>;

    /// Sets the channel layout of the audio buffer.
    fn set_channel_layout(&mut self, channel_layout: AudioChannelLayout);

    /// Processes an audio buffer in-place, applying high-pass then low-pass.
    fn process(&mut self, input: &mut AudioBuffer<T>);

    /// Updates the low cutoff frequency and resets the high-pass filter state.
    fn set_low_cut(&mut self, low_cut: f32);

    /// Gets the current low cutoff frequency.
    fn get_low_cut(&self) -> f32;

    /// Updates the high cutoff frequency and resets the low-pass filter state.
    fn set_high_cut(&mut self, high_cut: f32);

    /// Gets the current high cutoff frequency.
    fn get_high_cut(&self) -> f32;

    /// Updates the sample rate and resets the low-pass filter state.
    fn set_sample_rate(&mut self, sample_rate: f32);

    /// Gets the current sample rate.
    fn get_sample_rate(&self) -> f32;
}

pub trait AudioMathPrivateImpl<T: FloatType>: Sized {
    fn db_to_gain(db: T) -> T;
    fn gain_to_db(gain: T) -> T;
}

pub trait AudioFilterPrivateImpl<T: FloatType + Float>: Sized {
    fn process_channel(&mut self, input: &mut [T], channel: usize);
}

/// Bandpass filter combining high-pass and low-pass biquad filters
pub struct AudioFilter<T: FloatType + Float> {
    pub(super) lowpass: Vec<DirectForm1<T>>,
    pub(super) highpass: Vec<DirectForm1<T>>,
    pub(super) low_cut: f32,
    pub(super) high_cut: f32,
    pub(super) sample_rate: f32,
    pub(super) channel_layout: AudioChannelLayout,
}

pub struct AudioMath {}
