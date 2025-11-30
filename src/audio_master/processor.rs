use std::{
    cell::RefCell,
    rc::Rc,
    sync::{Arc, RwLock},
};

use fxhash::FxHashSet;
use num_traits::Float;

use crate::{
    addons::VecMove,
    audio_buffer::{AudioBuffer, AudioBufferImpl, AudioChannelLayout},
    audio_math::{AudioFilter, AudioFilterImpl},
    consts::{DEFAULT_AUDIO_MIX, MAX_AUDIO_MIX, MAX_SAMPLE_RATE, MIN_AUDIO_MIX, MIN_SAMPLE_RATE},
    float_type::FloatType,
};

use super::audio_master::{
    AudioAnalyser, AudioEffect, AudioEffectImpl, AudioEffectPrivateImpl, AudioEffectState,
    AudioProcessor, AudioProcessorImpl, AudioProcessorInner, AudioProcessorPrivateImpl,
};

impl<T: FloatType + Float> AudioProcessorImpl<T> for AudioProcessor<T> {
    fn new(sample_rate: f32, buffer_size: usize) -> Self {
        let sample_rate = sample_rate.clamp(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);

        let inner = AudioProcessorInner {
            // input: AudioBuffer::new(buffer_size, AudioChannelLayout::Stereo),
            sample_rate: sample_rate,
            buffer: AudioBuffer::new(buffer_size, AudioChannelLayout::Stereo),
            effects: FxHashSet::default(),
            effects_seq: Vec::new(),
            filter: AudioFilter::new(1.0, sample_rate, sample_rate, AudioChannelLayout::Stereo),
            use_filter: false,
            mix: DEFAULT_AUDIO_MIX,
            analyser: None,
        };

        return Self {
            inner: Arc::new(RwLock::new(inner)),
        };
    }

    #[inline]
    fn set_mix(&mut self, mix: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner.mix = mix.clamp(MIN_AUDIO_MIX, MAX_AUDIO_MIX);
        }
    }

    #[inline]
    fn get_mix(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.mix;
        }
    }

    #[inline]
    fn use_filter(&mut self, use_filter: bool) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.use_filter = use_filter;
        }
    }

    #[inline]
    fn is_using_filter(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.use_filter;
        }
    }

    #[inline]
    fn set_low_cut(&mut self, low_cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.filter.set_low_cut(low_cut);
        }
    }

    #[inline]
    fn set_high_cut(&mut self, high_cut: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.filter.set_high_cut(high_cut);
        }
    }

    #[inline]
    fn get_low_cut(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.filter.get_low_cut();
        }
    }

    #[inline]
    fn get_high_cut(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.filter.get_high_cut();
        }
    }

    #[inline]
    fn set_sample_rate(&mut self, sample_rate: f32) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            if inner.sample_rate == sample_rate {
                return;
            }
            let sample_rate = sample_rate.clamp(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);
            inner.sample_rate = sample_rate;
            inner.set_effects_sample_rate(sample_rate);
        }
    }

    #[inline]
    fn get_sample_rate(&self) -> f32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.sample_rate;
        }
    }

    fn add_effect(&mut self, mut effect: AudioEffect<T>) -> AudioEffectState {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            {
                effect
                    .inner
                    .write()
                    .unwrap_unchecked()
                    .set_sample_rate(inner.sample_rate);
            }
            let name = effect.get_name();
            if inner.effects.contains(name) {
                return AudioEffectState::EffectAlreadyExists;
            }

            inner.effects.insert(name.to_string());

            inner.effects_seq.push(effect);

            return AudioEffectState::EffectOk;
        }
    }

    fn remove_effect(&mut self, name: &str) -> Result<AudioEffect<T>, AudioEffectState> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            if !inner.effects.contains(name) {
                return Err(AudioEffectState::EffectNoEntry);
            }

            let index = inner
                .effects_seq
                .iter()
                .position(|x| x.get_name() == name)
                .unwrap();

            let effect = inner.effects_seq.remove(index);

            inner.effects.remove(name);

            return Ok(effect);
        }
    }

    #[inline]
    fn move_effect(&mut self, from: usize, to: usize) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.effects_seq.move_index(from, to);
        }
    }

    fn get_effect(&self, name: &str) -> Result<AudioEffect<T>, AudioEffectState> {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            if !inner.effects.contains(name) {
                return Err(AudioEffectState::EffectNoEntry);
            }

            let index = inner
                .effects_seq
                .iter()
                .position(|x| x.get_name() == name)
                .unwrap();

            unsafe {
                return Ok(inner.effects_seq.get_unchecked(index).clone());
            }
        }
    }

    #[inline]
    fn connect_analyser(&mut self, anal: Rc<RefCell<AudioAnalyser<T>>>) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            inner.analyser = Some(anal);
        }
    }

    #[inline]
    fn disconnect_analyser(&mut self) -> Option<Rc<RefCell<AudioAnalyser<T>>>> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            let anal = inner.analyser.take();

            return anal;
        }
    }
}
