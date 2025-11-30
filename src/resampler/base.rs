// Safe libsamplerate wrapper

use std::{
    any::TypeId,
    ffi::{CStr, c_double, c_int, c_long},
    marker::PhantomData,
    ptr,
};

use crate::{
    audio_buffer::AudioChannelLayout, consts::{MAX_RESAMPLE_RATIO, MIN_RESAMPLE_RATIO}, ffi::libsamplerate::{
        SRC_CONVERTER_TYPE, SRC_DATA, SRC_STATE, src_delete, src_new, src_process, src_reset,
        src_strerror,
    }, float_type::FloatType
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResamplerQuality {
    /// SRC_CONVERTER_TYPE::SRC_LINEAR
    Linear,

    /// SRC_CONVERTER_TYPE::SRC_SINC_FASTEST
    SincLow,

    /// SRC_CONVERTER_TYPE::SRC_SINC_MEDIUM_QUALITY
    SincMedium,

    /// SRC_CONVERTER_TYPE::SRC_SINC_BEST_QUALITY
    SincHigh,
}

/* Public Resampler API */
pub trait ResamplerImpl<T: FloatType>: Sized {
    fn new(quality: ResamplerQuality, layout: AudioChannelLayout) -> Resampler<T>;

    /// Accept interleaved buffers
    fn process(&mut self, input: &[T], output: &mut [T], src_ratio: f64) -> usize;

    /// Set other ResamplerQuality.
    /// Reallocates SRC_STATE
    fn set_quality(&mut self, quality: ResamplerQuality);

    /// ResamplerQuality getter.
    fn get_quality(&self) -> ResamplerQuality;

    fn get_channel_layout(&self) -> AudioChannelLayout;
    fn set_channel_layout(&mut self, layout: AudioChannelLayout);

    /// Panics if a reset error occured.
    fn reset(&mut self);
}

#[cfg(feature = "libsamplerate")]
pub struct Resampler<T: FloatType> {
    /// Drops in ResamplerPrivateImpl::_drop();
    pub(super) state: *mut SRC_STATE,

    pub(super) ch_layout: AudioChannelLayout,
    pub(super) quality: ResamplerQuality,

    pub(super) _phantom: PhantomData<T>,
}

#[cfg(not(feature = "libsamplerate"))]
pub struct Resampler<T: FloatType> {
    pub(super) ch_layout: AudioChannelLayout,

    pub(super) _ph: PhantomData<T>,
    pub(super) kernel_size: usize,
    pub(super) window: Vec<T>,
    pub(super) quality: ResamplerQuality
}

#[cfg(not(feature = "libsamplerate"))]
/// Kernel size
pub enum QTable {
    Linear = 16,
    SincLow = 32,
    SincMedium = 64,
    SincHigh = 128,
}