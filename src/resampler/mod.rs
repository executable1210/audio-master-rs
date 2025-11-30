mod base;

#[cfg(feature = "libsamplerate")]
mod resampler_libsamplerate;

#[cfg(not(feature = "libsamplerate"))]
mod resampler_sinc;

pub use base::{Resampler, ResamplerImpl, ResamplerQuality};