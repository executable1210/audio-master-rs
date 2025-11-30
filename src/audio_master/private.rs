use std::{
    cell::RefCell,
    fmt::format,
    marker::PhantomData,
    rc::Rc,
    sync::{Arc, RwLock},
};

use num_traits::Float;
use rustfft::num_complex::Complex;

use crate::{
    audio_buffer::{
        AudioBuffer, AudioBufferImpl, AudioBufferInterleaved, AudioBufferInterleavedImpl,
        AudioBufferMathImpl, AudioChannelLayout,
    },
    audio_math::{AudioFilter, AudioFilterImpl},
    backend::{AudioHost, AudioHostImpl, Device, RawAudioStreamState, StreamFeederTrait},
    consts::{
        DEFAULT_FRAME_RATE, DEFAULT_PROCESSOR_INPUT_NAME, DEFAULT_PROCESSOR_OUTPUT_NAME,
        DEFAULT_RESAMPLER_QUALITY, DEFAULT_SPEED, DEFAULT_VOLUME, MAX_AUDIO_MIX,
        MAX_RESAMPLE_SPEED, MAX_SAMPLE_RATE, MAX_VOLUME, MIN_AUDIO_MIX, MIN_RESAMPLE_SPEED,
        MIN_SAMPLE_RATE, MIN_VOLUME,
    },
    float_type::FloatType,
    globals::{G_BUFFER_SIZE, G_DEVICE_SAMPLE_RATE, G_FRAME_RATE},
    resampler::{Resampler, ResamplerImpl},
};

use super::audio_master::{
    AudioAnalyser, AudioAnalyserPrivateImpl, AudioAnalysis, AudioAnalysisPrivateImpl,
    AudioEffectImpl, AudioEffectInner, AudioEffectPrivateImpl, AudioEffectState, AudioMasterInner,
    AudioMasterPrivateImpl, AudioProcessor, AudioProcessorImpl, AudioProcessorInner,
    AudioProcessorPrivateImpl, AudioStreamContext, AudioStreamFeederTrait, AudioStreamInner,
    AudioStreamPrivateImpl, AudioStreamState, EffectParamType, EffectParamValue, MainStreamFeeder,
};

impl AudioMasterPrivateImpl for AudioMasterInner {
    fn try_to_create_default(
        buffer_size: Option<usize>,
        frame_rate: Option<usize>,
    ) -> Option<Arc<RwLock<Self>>> {
        let host = AudioHost::new();
        let device = host.default_output_device()?;

        return Some(AudioMasterInner::factory(
            host.clone(),
            device,
            buffer_size,
            frame_rate,
        ));
    }

    fn factory(
        mut host: AudioHost,
        device: Device,
        buffer_size: Option<usize>,
        frame_rate: Option<usize>,
    ) -> Arc<RwLock<Self>> {
        unsafe {
            G_DEVICE_SAMPLE_RATE = device.sample_rate as f32;
            if let Some(buffer_size) = buffer_size {
                if buffer_size == 0 {
                    panic!("Buffer size is 0");
                }
                G_FRAME_RATE = DEFAULT_FRAME_RATE;
                G_BUFFER_SIZE = buffer_size;
            } else {
                let frame_rate = frame_rate.unwrap();
                if frame_rate == 0 {
                    panic!("Frame rate is 0");
                }
                G_FRAME_RATE = frame_rate;
                G_BUFFER_SIZE = device.sample_rate as usize / frame_rate;
            }

            let master = Arc::new(RwLock::new(AudioMasterInner {
                host: host.clone(),
                current_device: Some(device),
                streams_f32: Vec::new(),
                streams_f64: Vec::new(),
                main_stream: None,
            }));

            let feeder: Box<dyn StreamFeederTrait<f32>> = Box::new(MainStreamFeeder {
                master: master.clone(),
                _phantom: PhantomData::default(),
            });

            let main_stream = match host.create_stream_default(feeder, G_BUFFER_SIZE) {
                Ok(x) => x,
                Err(x) => {
                    panic!("Couldn't create main stream: {:?}", x);
                }
            };

            master.write().unwrap_unchecked().main_stream = Some(main_stream);

            return master;
        }
    }
}

impl StreamFeederTrait<f32> for MainStreamFeeder<f32> {
    fn process(&mut self, input: &mut AudioBuffer<f32>) -> RawAudioStreamState {
        unsafe {
            input.reset();
            let mut master = self.master.write().unwrap();
            if master.streams_f32.is_empty() && master.streams_f64.is_empty() {
                return RawAudioStreamState::Silence;
            }

            for (buffer, stream) in master.streams_f32.iter_mut() {
                stream.write().unwrap_unchecked().process(buffer);
                input.mix_with(&buffer);
            }

            for (buffer, stream) in master.streams_f64.iter_mut() {
                stream.write().unwrap_unchecked().process(buffer);
                let mut converted = buffer.to_f32();
                input.mix_with(&mut converted);
            }

            return RawAudioStreamState::Ok;
        }
    }

    fn emit_stream_close(&mut self) {}

    fn emit_stream_error(&mut self, error: &str) {}

    fn emit_stream_pause(&mut self) {}

    fn emit_stream_play(&mut self) {}
}

impl<T: FloatType + Float + 'static> AudioStreamPrivateImpl<T> for AudioStreamInner<T> {
    fn factory(
        feeder: Box<dyn AudioStreamFeederTrait<T>>,
        sample_rate: u32,
        channel_layout: AudioChannelLayout,
    ) -> Self {
        unsafe {
            return Self {
                feeder,
                sample_rate: sample_rate as f32,
                speed: DEFAULT_SPEED,
                volume: DEFAULT_VOLUME,
                resample_factor: 1.0,
                channel_layout,
                buffer_resampled: AudioBufferInterleaved::new(G_BUFFER_SIZE * 12, channel_layout),
                buffer: AudioBufferInterleaved::new(G_BUFFER_SIZE, channel_layout),
                cb_buffer: AudioBuffer::new(G_BUFFER_SIZE, channel_layout),
                resampled_buffer_pos: 0,
                resampled_buffer_len: 0,
                use_filter: false,
                use_normalization: false,
                use_processor: true,
                processor: AudioProcessor::new(sample_rate as f32, G_BUFFER_SIZE),
                filter: AudioFilter::new(1000.0, 22050.0, sample_rate as f32, channel_layout),
                resampler: Resampler::new(DEFAULT_RESAMPLER_QUALITY, channel_layout),
                timestamp: 0.0,
                is_paused: true,
            };
        }
    }

    fn process(&mut self, input: &mut AudioBuffer<T>) {
        unsafe {
            input.reset();
            if self.is_paused {
                return;
            }

            assert!(
                input.channel_layout() == self.channel_layout,
                "Input channel layout ({:?}) does not match instance channel layout ({:?})",
                input.channel_layout(),
                self.channel_layout
            );

            let mut input_pos = 0;
            while input_pos < input.len() {
                if self.resampled_buffer_pos >= self.resampled_buffer_len {
                    let context = AudioStreamContext {
                        timestamp: self.timestamp,
                        sample_rate: self.sample_rate as u32,
                    };

                    let state = self.feeder.process(&context, &mut self.cb_buffer);
                    if state == AudioStreamState::Err {
                        input.reset();
                        return;
                    } else if state == AudioStreamState::Silence {
                        input.reset();
                        return;
                    }

                    self.pre_process();

                    self.cb_buffer
                        .copy_to_interleaved(self.buffer.get_buffer_mut());

                    let speed = self.speed.clamp(MIN_RESAMPLE_SPEED, MAX_RESAMPLE_SPEED);
                    let input_sample_rate =
                        G_DEVICE_SAMPLE_RATE.clamp(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);
                    let sample_rate = self.sample_rate.clamp(MIN_SAMPLE_RATE, MAX_SAMPLE_RATE);

                    let resample_ratio = ((sample_rate / input_sample_rate) + speed - 1.0) as f64;

                    let process_sample_rate: f32 =
                        (input_sample_rate * resample_ratio as f32) as f32;

                    self.processor.set_sample_rate(process_sample_rate);

                    self.resample_factor = resample_ratio;
                    self.buffer_resampled.reset();
                    self.resampled_buffer_len = self.resampler.process(
                        self.buffer.get_buffer(),
                        self.buffer_resampled.get_buffer_mut(),
                        resample_ratio as f64,
                    ) as usize;

                    self.resampled_buffer_pos = 0;

                    assert!(self.resampled_buffer_len != 0, "Resampling failed");
                }

                let frames_to_copy = (self.resampled_buffer_len - self.resampled_buffer_pos)
                    .min(input.len() - input_pos);

                input.copy_from_interleaved_range(
                    &self.buffer_resampled.get_buffer(),
                    self.resampled_buffer_pos,
                    self.resampled_buffer_pos + frames_to_copy,
                    input_pos,
                );

                input_pos += frames_to_copy;
                self.resampled_buffer_pos += frames_to_copy;
            }

            self.post_process(input);
        }
    }

    fn pre_process(&mut self) {
        if self.use_filter {
            self.filter.set_sample_rate(self.sample_rate);
            self.filter.process(&mut self.cb_buffer);
        }
    }

    fn post_process(&mut self, input: &mut AudioBuffer<T>) {
        unsafe {
            if self.use_processor {
                let mut proc = self.processor.inner.write().unwrap_unchecked();
                proc.set_buffer_size(self.buffer.len_planar());
                proc.process(input);
            }

            for i in 0..input.channel_layout() as usize {
                let ch = input.get_channel_mut(i);

                let volume = self.volume.clamp(MIN_VOLUME, MAX_VOLUME);

                for sample in ch.iter_mut() {
                    *sample = *sample * volume.into();
                }
            }

            if self.use_normalization {
                input.normalize_rms(1.0.into());
            }

            self.timestamp += input.len() as f64 * self.resample_factor;
        }
    }

    fn clear_buffers(&mut self) {
        self.buffer.reset();
        self.cb_buffer.reset();
        self.buffer_resampled.reset();

        self.resampled_buffer_pos = 0;
        self.resampled_buffer_len = 0;
    }

    fn set_buffer_size(&mut self, new_buff_size: usize) {
        self.buffer.resize(new_buff_size);
        self.cb_buffer.resize(new_buff_size);
        self.buffer_resampled.resize(new_buff_size * 8);
        self.processor
            .inner
            .write()
            .unwrap()
            .set_buffer_size(new_buff_size);

        self.resampled_buffer_pos = 0;
        self.resampled_buffer_len = 0;
    }
}

impl<'a> EffectParamValue<'a> {
    pub(super) fn to_type(&self) -> EffectParamType {
        match self {
            EffectParamValue::U8(_) => EffectParamType::U8,
            EffectParamValue::U16(_) => EffectParamType::U16,
            EffectParamValue::U32(_) => EffectParamType::U32,
            EffectParamValue::U64(_) => EffectParamType::U64,

            EffectParamValue::I8(_) => EffectParamType::I8,
            EffectParamValue::I16(_) => EffectParamType::I16,
            EffectParamValue::I32(_) => EffectParamType::I32,
            EffectParamValue::I64(_) => EffectParamType::I64,

            EffectParamValue::F32(_) => EffectParamType::F32,
            EffectParamValue::F64(_) => EffectParamType::F64,

            EffectParamValue::Str(_) => EffectParamType::Str,
        }
    }
}

impl AudioAnalysisPrivateImpl for AudioAnalysis {
    fn new(len: usize) -> Self {
        unsafe {
            let mut analysis: Vec<f32> = Vec::with_capacity(len);
            analysis.set_len(len);

            Self {
                name: String::default(),
                avg_volume_db: 0.0,
                clippings: 0,
                clipping_ratio: 0.0,
                analysis,
            }
        }
    }
}

impl<T: FloatType> AudioAnalyserPrivateImpl<T> for AudioAnalyser<T> {
    #[inline]
    fn resize(&mut self, new_len: usize) {
        self.mixed.resize(new_len, 0.0.into());
        self.signal.resize(new_len, Complex::new(0.0, 0.0));
    }

    fn calc_volume_and_clippings(&mut self) {
        unsafe {
            let mut clippings = 0;
            let mut sum_sq: T = 0.0.into();

            for sample in self.mixed.iter() {
                if *sample > 1.0.into() || *sample < (-1.0).into() {
                    clippings += 1;
                }
                sum_sq += *sample * *sample;
            }

            let rms = (sum_sq / (self.mixed.len() as f32).into()).sqrt();
            let rms_db = (rms / 1.0.into()).log10() * 20.0.into(); // dBFS (0 dBFS = full-scale sine)

            let clipping_ratio = clippings as f32 / self.mixed.len() as f32 * 100.0;

            let anal = self.anals.get_unchecked_mut(self.seq_index);

            anal.avg_volume_db = rms_db.to_f32();
            anal.clipping_ratio = clipping_ratio;
            anal.clippings = clippings;
        }
    }

    fn analyze(&mut self) {
        unsafe {
            let fft = self.planner.plan_fft_forward(self.mixed.len());

            for (cmx, sample) in self.mixed.iter().zip(self.signal.iter_mut()) {
                *sample = Complex::new((*cmx).to_f32(), 0.0);
            }

            fft.process(&mut self.signal);

            let anal = self.anals.get_unchecked_mut(self.seq_index);

            for (mag, cmx) in anal.analysis.iter_mut().zip(self.signal.iter()) {
                *mag = cmx.norm();
            }
        }
    }

    fn start_sequence(&mut self, size: usize) {
        assert!(self.seq_started, "Process sequence is already started");
        if size == self.seq_size {
            self.seq_index = 0;
            self.seq_started = true;
            return;
        }
        self.seq_size = size;
        self.anals.resize_with(size, || AudioAnalysis::new(size));
        self.seq_index = 0;
        self.seq_started = true;
    }

    fn end_sequence(&mut self) {
        (self.user_cb)(&self.anals);
        self.seq_index = 0;
        self.seq_started = false;
    }

    fn process_next_node(&mut self, name: &str, input: &AudioBuffer<T>) {
        unsafe {
            assert!(self.seq_index < self.seq_size, "Too much for this one");
            assert!(input.len() == self.mixed.len(), "Buffers size don't match");

            input.copy_to_interleaved(&mut self.mixed);
            self.calc_volume_and_clippings();

            let anal = self.anals.get_unchecked_mut(self.seq_index);

            if anal.name != name {
                anal.name = name.to_string();
            }

            self.seq_index += 1;
        }
    }
}

impl<T: FloatType + Float> AudioEffectPrivateImpl<T> for AudioEffectInner<T> {
    fn process(&mut self, input: &mut AudioBuffer<T>) -> AudioEffectState {
        unsafe {
            self.buffer.copy_from(&input);

            let buffer = &mut self.buffer as *mut AudioBuffer<T>;

            if self.use_filter_in {
                self.filter_in.process(&mut (*buffer));
            }

            let state = if self.is_muffled {
                self.buffer.reset();
                self.processor.process(&mut (*buffer))
            } else {
                self.processor.process(&mut (*buffer))
            };

            match state {
                AudioEffectState::ProcessOk => {}
                AudioEffectState::ProcessError => {
                    return AudioEffectState::ProcessError;
                }
                _ => {}
            }

            if self.use_filter_out {
                self.filter_out.process(&mut (*buffer));
            }

            let mix = self.mix.clamp(MIN_AUDIO_MIX, MAX_AUDIO_MIX);
            input.mix_with_gain(&self.buffer, mix);

            return AudioEffectState::ProcessOk;
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        unsafe {
            self.processor.set_sample_rate(sample_rate);
            self.sample_rate = sample_rate;
            self.filter_in.set_sample_rate(sample_rate);
            self.filter_out.set_sample_rate(sample_rate);
        }
    }

    fn resize(&mut self, new_len: usize) {
        unsafe {
            self.buffer.resize(new_len);
        }
    }
}

impl<T: FloatType + Float> AudioProcessorPrivateImpl<T> for AudioProcessorInner<T> {
    #[inline]
    fn set_buffer_size(&mut self, new_len: usize) {
        unsafe {
            self.buffer.resize(new_len);

            for effect in self.effects_seq.iter_mut() {
                effect.inner.write().unwrap_unchecked().resize(new_len);
            }

            if let Some(anal) = &self.analyser {
                let mut anal = anal.borrow_mut();

                anal.resize(new_len);
            }
        }
    }

    #[inline]
    fn set_effects_sample_rate(&mut self, sample_rate: f32) {
        unsafe {
            for fx in self.effects_seq.iter_mut() {
                fx.inner
                    .write()
                    .unwrap_unchecked()
                    .set_sample_rate(sample_rate);
            }
        }
    }

    #[inline]
    fn anal_start_sequence(anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>, size: usize) {
        if let Some(anal) = anal {
            anal.borrow_mut().start_sequence(size);
        }
    }

    #[inline]
    fn anal_end_sequence(anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>) {
        if let Some(anal) = anal {
            anal.borrow_mut().end_sequence();
        }
    }

    #[inline]
    fn analyze_next_node(
        anal: &Option<Rc<RefCell<AudioAnalyser<T>>>>,
        name: &str,
        input: &AudioBuffer<T>,
    ) {
        if let Some(anal) = anal {
            anal.borrow_mut().process_next_node(name, input);
        }
    }

    fn process(&mut self, input: &mut AudioBuffer<T>) -> AudioEffectState {
        unsafe {
            let orig_layout = input.channel_layout();
            input.set_channel_layout(AudioChannelLayout::Stereo);
            AudioProcessorInner::anal_start_sequence(&self.analyser, self.effects_seq.len() + 2); // 2 is for input and output

            self.buffer.copy_from(&input);
            AudioProcessorInner::analyze_next_node(
                &self.analyser,
                DEFAULT_PROCESSOR_INPUT_NAME,
                &self.buffer,
            );

            // to prevent borrow checker
            let mut buffer = (&self.buffer as *const AudioBuffer<T>) as *mut AudioBuffer<T>;
            let mut analyser = (&self.analyser as *const Option<Rc<RefCell<AudioAnalyser<T>>>>)
                as *mut Option<Rc<RefCell<AudioAnalyser<T>>>>;

            for fx in self.effects_seq.iter_mut() {
                fx.inner.write().unwrap_unchecked().process(&mut (*buffer));

                AudioProcessorInner::analyze_next_node(
                    &mut (*analyser),
                    fx.get_name(),
                    &mut (*buffer),
                );
            }

            if self.use_filter {
                self.filter.process(&mut (*buffer));
            }

            let mix = self.mix.clamp(MIN_AUDIO_MIX, MAX_AUDIO_MIX);
            input.mix_with_gain(&(*buffer), mix);

            AudioProcessorInner::analyze_next_node(
                &self.analyser,
                DEFAULT_PROCESSOR_OUTPUT_NAME,
                &input,
            );

            input.set_channel_layout(orig_layout);

            AudioProcessorInner::anal_end_sequence(&self.analyser);

            return AudioEffectState::ProcessOk;
        }
    }
}
