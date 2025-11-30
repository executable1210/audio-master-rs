use num_traits::Float;

use crate::{
    audio_buffer::{AudioBufferImpl, AudioBufferInterleavedImpl, AudioChannelLayout},
    audio_math::AudioFilterImpl,
    consts::{MAX_RESAMPLE_SPEED, MAX_SAMPLE_RATE, MAX_VOLUME, MIN_RESAMPLE_SPEED, MIN_SAMPLE_RATE, MIN_VOLUME},
    float_type::FloatType,
    resampler::{ResamplerImpl, ResamplerQuality},
};

use super::audio_master::{AudioProcessor, AudioStream, AudioStreamImpl, AudioStreamPrivateImpl};

impl<T: FloatType + Float + 'static> AudioStreamImpl<T> for AudioStream<T> {
    fn use_normalization(&mut self, norm: bool) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.use_normalization = norm;
        }
    }
    fn is_using_normalization(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.use_normalization;
        }
    }

    fn get_timestamp(&self) -> f64 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.timestamp;
        }
    }

    fn get_resample_quality(&self) -> ResamplerQuality {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.resampler.get_quality();
        }
    }

    fn set_resample_quality(&mut self, quality: ResamplerQuality) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
        }
    }

    fn get_speed(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.speed;
        }
    }

    fn set_speed(&mut self, speed: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.speed = speed.clamp(MIN_RESAMPLE_SPEED, MAX_RESAMPLE_SPEED)
        }
    }

    fn get_volume(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.volume;
        }
    }

    fn set_volume(&self, volume: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.volume = volume.clamp(MIN_VOLUME, MAX_VOLUME);
        }
    }

    fn use_filter(&mut self, filter: bool) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.use_filter = filter;
        }
    }

    fn is_using_filter(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.use_filter;
        }
    }

    fn get_low_cut(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.filter.get_low_cut();
        }
    }

    fn set_low_cut(&mut self, low_cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.filter.set_low_cut(low_cut);
        }
    }

    fn get_high_cut(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.filter.get_high_cut();
        }
    }

    fn set_high_cut(&mut self, high_cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.filter.set_high_cut(high_cut);
        }
    }

    fn get_sample_rate(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.sample_rate;
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.sample_rate = sample_rate.clamp(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);
        }
    }

    fn get_processor(&self) -> AudioProcessor<T> {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.processor.clone();
        }
    }

    fn resume(&mut self) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.is_paused = false;
        }
    }

    fn pause(&mut self) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.is_paused = true;
        }
    }

    fn get_channel_layout(&self) -> AudioChannelLayout {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.channel_layout;
        }
    }

    fn set_channel_layout(&mut self, layout: AudioChannelLayout) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            if inner.channel_layout == layout {
                return;
            }
            inner.channel_layout = layout;

            inner.buffer.set_channel_layout(layout);
            inner.cb_buffer.set_channel_layout(layout);
            inner.buffer_resampled.set_channel_layout(layout);
            inner.filter.set_channel_layout(layout);
            inner.resampler.set_channel_layout(layout);

            inner.clear_buffers();
        }
    }
}
