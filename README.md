# audio-master-rs

Cross-platform rich Audio API for Rust based on [cpal](https://example.com) and high-quality C library [libsamplerate](https://github.com/libsndfile/libsamplerate).

> **Status:** The project is not stable yet.

## Features

* Built-in `AudioProcessor` API for wrapping `Audio FX`'s or `DSP`
* Autohandler of different channel count for a specific device. 
* In-runtime `AudioStream`s layer `(Doesn't depend on [cpal](https://example.com) stream, there's only one main stream for handling `AudioStream`)`.
* High-Quality Resample algorithm. `Turn on libsamplerate feature`

## Requirements

* C/C++ Compiler for building [libsamplerate](https://github.com/libsndfile/libsamplerate)

## Quick example

Sine wave example
```rust
use audio_master::{
    AudioBuffer, AudioBufferImpl, AudioChannelLayout, AudioMaster, AudioMasterError,
    AudioMasterImpl, AudioStreamContext, AudioStreamFeederTrait, AudioStreamImpl,
    AudioStreamSettings, AudioStreamState,
};

const SAMPLE_RATE: u32 = 44100;
const SINEWAVE_FREQ: f64 = 1200.0;

struct SineWave {
    freq: f64,
    t: f64,
}

impl AudioStreamFeederTrait<f64> for SineWave {
    fn process(
        &mut self,
        context: &AudioStreamContext,
        input: &mut AudioBuffer<f64>,
    ) -> AudioStreamState {
        let ch_count = input.channel_layout() as usize;

        let sample_rate = context.sample_rate;

        for s_idx in 0..input.len() {
            let sample = (self.t * self.freq * 2.0 * std::f64::consts::PI).sin();
            self.t += 1.0 / sample_rate as f64;

            for ch_idx in 0..ch_count {
                let channel = unsafe { input.get_channel_unchecked_mut(ch_idx) };

                unsafe { *channel.get_unchecked_mut(s_idx) = sample }
            }
        }

        return AudioStreamState::Ok;
    }
}

fn main() -> Result<(), AudioMasterError> {
    // buffer_size = device sample rate / fps
    let mut master = AudioMaster::new_with_fps(25);

    // start main system stream
    let _ = master.start_sys_stream()?;

    let settings = AudioStreamSettings {
        sample_rate: SAMPLE_RATE,
        channel_layout: AudioChannelLayout::Stereo,
    };

    let sine_feeder: Box<dyn AudioStreamFeederTrait<f64>> = Box::new(SineWave {
        freq: SINEWAVE_FREQ,
        t: 0.0,
    });

    let mut stream = master.create_stream_f64(&settings, sine_feeder);

    // start process stream
    stream.resume();

    std::thread::sleep(std::time::Duration::from_secs(16));

    return Ok(());
}
```