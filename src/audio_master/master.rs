use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use crate::{
    audio_buffer::{AudioBuffer, AudioBufferImpl},
    audio_master::audio_master::AudioStreamPrivateImpl,
    backend::{AudioHost, AudioHostImpl, Device, RawAudioStreamImpl, StreamFeederTrait},
    consts::DEFAULT_FRAME_RATE,
    globals::{G_BUFFER_SIZE, G_DEVICE_SAMPLE_RATE},
};

pub use super::audio_master::{
    AudioMaster, AudioMasterError, AudioMasterImpl, AudioMasterInner, AudioMasterPrivateImpl,
    AudioStream, AudioStreamFeederTrait, AudioStreamInner, AudioStreamSettings, MainStreamFeeder,
};

impl Default for AudioMaster {
    fn default() -> Self {
        return AudioMaster {
            inner: Arc::new(RwLock::new(AudioMasterInner {
                host: AudioHost::new(),
                current_device: None,
                streams_f32: Vec::new(),
                streams_f64: Vec::new(),
                main_stream: None,
            })),
        };
    }
}

impl AudioMasterImpl for AudioMaster {
    fn new() -> AudioMaster {
        match AudioMasterInner::try_to_create_default(None, Some(DEFAULT_FRAME_RATE)) {
            Some(inner) => return AudioMaster { inner: inner },
            None => return AudioMaster::default(),
        };
    }

    fn new_with_buffer_size(buffer_size: usize) -> AudioMaster {
        match AudioMasterInner::try_to_create_default(Some(buffer_size), None) {
            Some(inner) => return AudioMaster { inner: inner },
            None => return AudioMaster::default(),
        };
    }

    fn new_with_fps(fps: usize) -> AudioMaster {
        match AudioMasterInner::try_to_create_default(None, Some(fps)) {
            Some(inner) => return AudioMaster { inner: inner },
            None => return AudioMaster::default(),
        };
    }

    fn stop_sys_stream(&mut self) -> Result<(), AudioMasterError> {
        unsafe {
            if let Some(stream) = self.inner.write().unwrap_unchecked().main_stream.as_mut() {
                stream.pause();
                return Ok(());
            }

            return Err(AudioMasterError::StreamNotInitialized);
        }
    }

    fn start_sys_stream(&mut self) -> Result<(), AudioMasterError> {
        unsafe {
            if let Some(stream) = self.inner.write().unwrap_unchecked().main_stream.as_mut() {
                stream.play();
                return Ok(());
            }

            return Err(AudioMasterError::StreamNotInitialized);
        }
    }

    fn try_to_initialize_stream(&mut self) -> Result<(), AudioMasterError> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            match inner.host.default_output_device() {
                Some(device) => {
                    G_DEVICE_SAMPLE_RATE = device.sample_rate as f32;
                    inner.current_device = Some(device);
                }
                None => {
                    inner.current_device = None;
                    return Err(AudioMasterError::NoDeviceAvailable);
                }
            }

            let feeder: Box<dyn StreamFeederTrait<f32>> = Box::new(MainStreamFeeder {
                master: self.inner.clone(),
                _phantom: PhantomData::default(),
            });

            match inner.host.create_stream_default(feeder, G_BUFFER_SIZE) {
                Ok(x) => {
                    inner.main_stream = Some(x);
                    return Ok(());
                }
                Err(x) => {
                    inner.current_device = None;
                    return Err(AudioMasterError::Internal(format!("{:?}", x)));
                }
            };
        }
    }

    fn devices(&self) -> Vec<Device> {
        return unsafe { self.inner.write().unwrap_unchecked().host.devices() };
    }

    fn get_current_device(&self) -> Option<Device> {
        return unsafe { self.inner.write().unwrap_unchecked().current_device.clone() };
    }

    fn change_device(&self, device_id: usize) -> Result<(), AudioMasterError> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            match inner.host.get_device_by_id(device_id) {
                Ok(device) => {
                    G_DEVICE_SAMPLE_RATE = device.sample_rate as f32;
                    inner.current_device = Some(device);
                }
                Err(e) => return Err(AudioMasterError::Internal(format!("{:?}", e))),
            }

            let feeder: Box<dyn StreamFeederTrait<f32>> = Box::new(MainStreamFeeder {
                master: self.inner.clone(),
                _phantom: PhantomData::default(),
            });

            match inner.host.create_stream(feeder, G_BUFFER_SIZE, device_id) {
                Ok(x) => {
                    inner.main_stream = Some(x);
                    return Ok(());
                }
                Err(x) => {
                    inner.current_device = None;
                    return Err(AudioMasterError::Internal(format!("{:?}", x)));
                }
            };
        }
    }

    fn create_stream_f32(
        &mut self,
        settings: &AudioStreamSettings,
        feeder: Box<dyn AudioStreamFeederTrait<f32>>,
    ) -> AudioStream<f32> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            let stream_inner = Arc::new(RwLock::new(AudioStreamInner::factory(
                feeder,
                settings.sample_rate,
                settings.channel_layout,
            )));

            inner.streams_f32.push((
                AudioBuffer::<f32>::new(G_BUFFER_SIZE, settings.channel_layout),
                stream_inner.clone(),
            ));

            return AudioStream {
                inner: stream_inner,
            };
        }
    }

    fn create_stream_f64(
        &mut self,
        settings: &AudioStreamSettings,
        feeder: Box<dyn AudioStreamFeederTrait<f64>>,
    ) -> AudioStream<f64> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();
            let stream_inner = Arc::new(RwLock::new(AudioStreamInner::factory(
                feeder,
                settings.sample_rate,
                settings.channel_layout,
            )));

            inner.streams_f64.push((
                AudioBuffer::<f64>::new(G_BUFFER_SIZE, settings.channel_layout),
                stream_inner.clone(),
            ));

            return AudioStream {
                inner: stream_inner,
            };
        }
    }
}
