// Windows/Linux/MacOS

use std::sync::{Arc, RwLock};

use cpal::traits::{HostTrait, StreamTrait};

use crate::{float_type::FloatType};

use super::backend::{
    AudioHost, AudioHostError, AudioHostImpl, AudioHostPrivateImpl, Device, Private, PrivateImpl,
    RawAudioStream, RawAudioStreamError, RawAudioStreamImpl, StreamFeederTrait,
};

impl AudioHostImpl for AudioHost {
    fn new() -> AudioHost {
        return Self { core: Arc::new(RwLock::new(cpal::Host::default())) }
    }

    fn default_output_device(&self) -> Option<Device> {
        unsafe {
            let host = self.core.read().unwrap_unchecked();

            match host.default_output_device() {
                Some(cpal_device) => {
                    return Private::cpal_device_to_device(cpal_device);
                }
                None => {
                    return None;
                }
            }
        }
    }

    fn get_device_by_id(&self, id: usize) -> Result<Device, AudioHostError> {
        let devices = self.devices();

        let device = devices.into_iter().find(|device| device.id == id);

        match device {
            Some(x) => {
                return Ok(x);
            }
            None => {
                return Err(AudioHostError::NoDevice);
            }
        }
    }

    fn devices(&self) -> Vec<Device> {
        unsafe {
            let host = self.core.read().unwrap_unchecked();

            match host.output_devices() {
                Ok(devices) => {
                    let devices: Vec<Device> = devices
                        .filter_map(|cpal_device| {
                            match Private::cpal_device_to_device(cpal_device) {
                                Some(device) => {
                                    return Some(device);
                                }
                                None => {
                                    return None;
                                }
                            }
                        })
                        .collect();

                    return devices;
                }
                Err(e) => {
                    return vec![];
                }
            }
        }
    }

    fn create_stream_default(
        &mut self,
        feeder: Box<dyn StreamFeederTrait<f32>>,
        buffer_size: usize,
    ) -> Result<RawAudioStream<f32>, AudioHostError> {
        unsafe {
            let host = self.core.clone();
            let host = host.write().unwrap_unchecked();
            let device = match host.default_output_device() {
                Some(x) => x,
                None => {
                    return Err(AudioHostError::NoDevice);
                }
            };

            let stream = self.create_stream_with_device(&device, buffer_size, feeder)?;

            return Ok(RawAudioStream { inner: stream });
        }
    }

    fn create_stream(
        &mut self,
        feeder: Box<dyn StreamFeederTrait<f32>>,
        buffer_size: usize,
        device_id: usize,
    ) -> Result<RawAudioStream<f32>, AudioHostError> {
        unsafe {
            let host = self.core.clone();
            let host = host.write().unwrap_unchecked();
            let device = match self.get_cpal_device_by_id(device_id) {
                Ok(x) => x,
                Err(e) => {
                    return Err(e);
                }
            };
            let stream = self.create_stream_with_device(&device, buffer_size, feeder)?;

            return Ok(RawAudioStream { inner: stream });
        }
    }
}

impl<T: FloatType> RawAudioStreamImpl for RawAudioStream<T> {
    /// Close stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_close().
    fn close(&mut self) {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            inner.core = None;
        }
    }

    /// Play stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_play().
    fn play(&mut self) -> Result<(), RawAudioStreamError> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            match inner.core.as_mut().unwrap_unchecked().play() {
                Ok(e) => return Ok(()),
                Err(e) => return Err(RawAudioStreamError::Internal(e.to_string())),
            }
        }
    }

    /// Pause stream.
    /// Can be emitted by StreamFeederTrait::emit_stream_pause().
    fn pause(&mut self) -> Result<(), RawAudioStreamError> {
        unsafe {
            let mut inner = self.inner.write().unwrap_unchecked();

            match inner.core.as_mut().unwrap_unchecked().pause() {
                Ok(e) => return Ok(()),
                Err(e) => return Err(RawAudioStreamError::Internal(e.to_string())),
            }
        }
    }

    /// Is stream running
    fn is_running(&self) -> bool {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.is_running;
        }
    }

    /// Returns stream sample rate
    fn get_sample_rate(&self) -> u32 {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return inner.sample_rate;
        }
    }

    /// Returns timestamp in micro seconds
    fn get_timestamp(&self) -> usize {
        unsafe {
            let inner = self.inner.read().unwrap_unchecked();
            return (inner.timestamp as f64 / inner.sample_rate_1000) as usize;
        }
    }
}
