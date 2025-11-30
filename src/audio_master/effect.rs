use std::sync::{Arc, RwLock};

use fxhash::FxHashMap;
use num_traits::Float;

use crate::{
    audio_buffer::{AudioBuffer, AudioBufferImpl, AudioChannelLayout},
    audio_math::{AudioFilter, AudioFilterImpl},
    consts::{
        DEFAULT_AUDIO_MIX, DEFAULT_SAMPLE_RATE, MAX_AUDIO_MIX, MAX_FREQ_CUT, MIN_AUDIO_MIX,
        MIN_FREQ_CUT,
    },
    float_type::FloatType,
};

use super::audio_master::{
    AudioEffect, AudioEffectImpl, AudioEffectInner, AudioEffectState, AudioEffectTrait,
    EffectParamType, EffectParamValue, ParamPropDesk,
};

impl<T: FloatType + Float> AudioEffectImpl<T> for AudioEffect<T> {
    fn new<const N: usize>(
        name: &str,
        param_props: ParamPropDesk<N>,
        processor: Box<dyn AudioEffectTrait<T>>,
    ) -> Self {
        let params: FxHashMap<String, EffectParamType> = param_props
            .into_iter()
            .map(|x| (x.0.to_string(), x.1))
            .collect();

        let inner = AudioEffectInner {
            params,
            name: name.to_string(),
            processor: processor,
            mix: DEFAULT_AUDIO_MIX,
            filter_in: AudioFilter::new(
                1.0,
                DEFAULT_SAMPLE_RATE,
                DEFAULT_SAMPLE_RATE,
                AudioChannelLayout::Stereo,
            ),
            filter_out: AudioFilter::new(
                1.0,
                DEFAULT_SAMPLE_RATE,
                DEFAULT_SAMPLE_RATE,
                AudioChannelLayout::Stereo,
            ),
            sample_rate: DEFAULT_SAMPLE_RATE,
            use_filter_in: false,
            use_filter_out: false,
            is_muffled: false,
            buffer: AudioBuffer::<T>::new_with_layout(AudioChannelLayout::Stereo),
        };
        AudioEffect {
            inner: Arc::new(RwLock::new(inner)),
        }
    }

    #[inline]
    fn use_filter_in(&mut self, use_filter: bool) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner.use_filter_in = use_filter;
        }
    }

    #[inline]
    fn use_filter_out(&mut self, use_filter: bool) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner.use_filter_out = use_filter;
        }
    }

    #[inline]
    fn is_using_filter_in(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();

            inner.use_filter_in
        }
    }

    #[inline]
    fn is_using_filter_out(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.use_filter_out
        }
    }

    #[inline]
    fn get_name(&self) -> &str {
        unsafe {
            // dangerous
            let inner = self.inner.read().unwrap_unchecked();
            let ptr = &inner.name as *const String;

            return (*ptr).as_str();
        }
    }

    #[inline]
    fn get_mix(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.mix
        }
    }

    #[inline]
    fn set_mix(&mut self, mix: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner.mix = mix.clamp(MIN_AUDIO_MIX, MAX_AUDIO_MIX);
        }
    }

    #[inline]
    fn set_low_cut_in(&mut self, cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner
                .filter_in
                .set_low_cut(cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT));
        }
    }

    #[inline]
    fn get_low_cut_in(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.filter_in.get_low_cut()
        }
    }

    #[inline]
    fn set_high_cut_in(&mut self, cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner
                .filter_in
                .set_high_cut(cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT));
        }
    }

    #[inline]
    fn get_high_cut_in(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.filter_in.get_high_cut()
        }
    }

    #[inline]
    fn set_low_cut_out(&mut self, cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner
                .filter_out
                .set_low_cut(cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT));
        }
    }

    #[inline]
    fn get_low_cut_out(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.filter_out.get_low_cut()
        }
    }

    #[inline]
    fn set_high_cut_out(&mut self, cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner
                .filter_out
                .set_high_cut(cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT));
        }
    }

    #[inline]
    fn get_high_cut_out(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            inner.filter_out.get_high_cut()
        }
    }

    #[inline]
    fn get_params(&self) -> &FxHashMap<String, EffectParamType> {
        unsafe {
            // dangerous
            let inner = self.inner.read().unwrap_unchecked();
            let ptr = &inner.params as *const FxHashMap<String, EffectParamType>;

            return &(*ptr);
        }
    }

    fn set_param(&mut self, name: &str, value: EffectParamValue) -> AudioEffectState {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            if !inner.params.contains_key(name) {
                return AudioEffectState::ParamNoEntry;
            }

            inner.processor.set_param(name, value);

            return AudioEffectState::EffectOk;
        }
    }

    fn get_param(&self, name: &str) -> Result<EffectParamValue, AudioEffectState> {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            let proc = (&inner.processor as *const Box<dyn AudioEffectTrait<T>>)
                as *mut Box<dyn AudioEffectTrait<T>>;
            match inner.params.get(name) {
                Some(x) => {
                    return (*proc).get_param(name);
                }
                None => {
                    return Err(AudioEffectState::ParamNoEntry);
                }
            }
        }
    }
}
