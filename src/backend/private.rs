use crate::{
    audio_buffer::{AudioBuffer, AudioBufferImpl, AudioChannelLayout},
    float_type::FloatType,
};
use cpal::{
    PauseStreamError, PlayStreamError, SampleFormat, SupportedStreamConfigRange,
    traits::{DeviceTrait, HostTrait},
};
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, RwLock},
};

use super::backend::{
    AudioHost, AudioHostError, AudioHostPrivateImpl, Device, Private, PrivateImpl,
    RawAudioStreamError, RawAudioStreamInner, RawAudioStreamPrivateImpl, RawAudioStreamState,
    StreamFeederTrait,
};

impl PrivateImpl for Private {
    fn channels_to_layout(channels: usize) -> Option<AudioChannelLayout> {
        match channels {
            1 => Some(AudioChannelLayout::Mono),
            2 => Some(AudioChannelLayout::Stereo),
            3 => Some(AudioChannelLayout::Surround30),
            4 => Some(AudioChannelLayout::Surround40),
            6 => Some(AudioChannelLayout::Surround51),
            8 => Some(AudioChannelLayout::Surround71),

            _ => None,
        }
    }

    #[inline(always)]
    fn conv_cpal_play_stream_error(error: PlayStreamError) -> RawAudioStreamError {
        return match error {
            PlayStreamError::DeviceNotAvailable => RawAudioStreamError::DeviceNotAvailable,
            PlayStreamError::BackendSpecific { err } => {
                RawAudioStreamError::Internal(err.to_string())
            }
        };
    }

    #[inline(always)]
    fn conv_cpal_pause_stream_error(error: PauseStreamError) -> RawAudioStreamError {
        return match error {
            PauseStreamError::DeviceNotAvailable => RawAudioStreamError::DeviceNotAvailable,
            PauseStreamError::BackendSpecific { err } => {
                RawAudioStreamError::Internal(err.to_string())
            }
        };
    }

    fn cpal_device_to_device(device: cpal::Device) -> Option<Device> {
        let name = match device.name() {
            Ok(name) => name,
            Err(e) => {
                return None;
            }
        };

        let _ = match device.supported_output_configs() {
            Ok(cfg) => cfg,
            Err(e) => {
                return None;
            }
        }; // just for checking

        let config = match device.default_output_config() {
            Ok(cfg) => cfg,
            Err(e) => {
                return None;
            }
        }; // just for checking

        let id = Private::generate_id_from_cpal_device(&device) as usize;

        let channels = Private::channels_to_layout(config.channels() as usize);

        if channels.is_none() {
            return None;
        }

        Some(Device {
            channels: channels.unwrap(),
            name: name,
            sample_rate: config.sample_rate().0,
            id: id,
        })
    }

    fn generate_id_from_cpal_device(device: &cpal::Device) -> i64 {
        let mut hasher = DefaultHasher::new();

        let name = device.name();
        if name.is_err() {
            return -1;
        }
        name.unwrap().hash(&mut hasher);

        let configs = device.supported_output_configs();

        if configs.is_err() {
            return -1;
        }
        let configs = configs.unwrap();

        for config in configs {
            Private::hash_config_range(&config, &mut hasher);
        }

        return hasher.finish() as i64;
    }

    fn hash_config_range(config: &SupportedStreamConfigRange, hasher: &mut DefaultHasher) {
        config.channels().hash(hasher);
        config.min_sample_rate().0.hash(hasher);
        config.max_sample_rate().0.hash(hasher);
    }
}

impl AudioHostPrivateImpl for AudioHost {
    fn get_cpal_device_by_id(&self, id: usize) -> Result<cpal::Device, AudioHostError> {
        unsafe {
            let host = self.core.write().unwrap_unchecked();

            let devices = host.devices();
            if devices.is_err() {
                let err = devices.err().unwrap();

                return Err(AudioHostError::Internal(err.to_string()));
            }
            let mut devices = devices.unwrap();

            let device = devices.find(|device| {
                let _id = Private::generate_id_from_cpal_device(&device);
                if _id == id as i64 {
                    return true;
                }
                return false;
            });

            match device {
                Some(x) => {
                    return Ok(x);
                }
                None => {
                    let err = AudioHostError::NoDevice;
                    return Err(err);
                }
            }
        }
    }

    fn create_stream_with_device<T: FloatType + Send + Sync + 'static>(
        &mut self,
        device: &cpal::Device,
        buffer_size: usize,
        feeder: Box<dyn StreamFeederTrait<T>>,
    ) -> Result<Arc<RwLock<RawAudioStreamInner<T>>>, AudioHostError> {
        unsafe {
            let config = match device.default_output_config() {
                Ok(x) => x,
                Err(e) => {
                    return Err(AudioHostError::NoConfig);
                }
            };

            if !config.sample_format().is_float() {
                return Err(AudioHostError::UnknownSampleFormat);
            }

            let channels = match Private::channels_to_layout(config.channels() as usize) {
                Some(x) => x,
                None => {
                    return Err(AudioHostError::UnknownChannelCount);
                }
            };

            let stream_config = config.config();

            let sample_rate = config.sample_rate().0;

            let stream = RawAudioStreamInner {
                core: None,
                is_running: false,
                buffer: AudioBuffer::new(buffer_size, channels),
                buffer_offset: buffer_size,
                sample_rate: sample_rate,
                sample_rate_1000: sample_rate as f64 / 1000.0,
                feeder: feeder,
                timestamp: 0,
            };

            let stream_thread = Arc::new(RwLock::new(stream));
            let stream_process = stream_thread.clone();
            let stream_err = stream_thread.clone();
            let cpal_stream = device.build_output_stream(
                &stream_config,
                move |buffer: &mut [f32], _info: &cpal::OutputCallbackInfo| {
                    let mut stream = stream_process.write().unwrap_unchecked();
                    stream.user_cpal_process(buffer, channels);
                },
                move |e| {
                    let mut stream = stream_err.write().unwrap_unchecked();

                    stream.feeder.emit_stream_error(&e.to_string());
                },
                None,
            );

            {
                let mut stream_write = stream_thread.write().unwrap();

                match cpal_stream {
                    Ok(x) => {
                        stream_write.core = Some(x);
                    }
                    Err(e) => return Err(AudioHostError::Internal(e.to_string())),
                }
            }

            return Ok(stream_thread);
        }
    }
}

unsafe impl<T: FloatType> Send for RawAudioStreamInner<T> {}
unsafe impl<T: FloatType> Sync for RawAudioStreamInner<T> {}

impl<T: FloatType + Send + Sync> RawAudioStreamPrivateImpl<T> for RawAudioStreamInner<T> {
    fn user_cpal_process(&mut self, buffer: &mut [f32], channels: AudioChannelLayout) {
        let channels = channels as usize;
        let total_frames = buffer.len() / channels;
        let mut frame_index = 0;

        let buffer_size = self.buffer.len();
        while frame_index < total_frames {
            if self.buffer_offset >= buffer_size {
                self.buffer_offset = 0;
                self.buffer.reset();
                let state = self.feeder.process(&mut self.buffer);

                if state == RawAudioStreamState::Err {
                    buffer.fill(0.0);
                    return;
                }

                let channels_type =
                    Private::channels_to_layout(channels).expect("Unknown count of channels");

                self.buffer.set_channel_layout(channels_type);
            }

            // How much frames should be copied
            let remaining_output_frames = total_frames - frame_index;
            let available_frames = buffer_size - self.buffer_offset;
            let frames_to_copy = remaining_output_frames.min(available_frames);

            unsafe {
                let buffer_raw = buffer.as_mut_ptr();
                for i in 0..frames_to_copy {
                    let base = (frame_index + i) * channels;

                    for ch in 0..channels {
                        let channel = self.buffer.get_channel_mut(ch);
                        let channel_raw = channel.as_ptr();

                        *buffer_raw.add(base + ch) =
                            (*channel_raw.add(self.buffer_offset + i)).to_f32();
                    }
                }
            }

            frame_index += frames_to_copy;
            self.buffer_offset += frames_to_copy;
        }
    }
}
