use std::{any::TypeId, marker::PhantomData};

use crate::{
    audio_buffer::AudioChannelLayout,
    consts::{MAX_RESAMPLE_RATIO, MIN_RESAMPLE_RATIO},
    float_type::FloatType,
};

use super::base::{QTable, Resampler, ResamplerImpl, ResamplerQuality};

impl ResamplerQuality {
    fn to_converter_type(&self) -> QTable {
        match self {
            ResamplerQuality::Linear => QTable::Linear,
            ResamplerQuality::SincLow => QTable::SincLow,
            ResamplerQuality::SincMedium => QTable::SincMedium,
            ResamplerQuality::SincHigh => QTable::SincHigh,
        }
    }
}


impl<T: FloatType + 'static> Resampler<T> {
    fn alloc_window(kernel_size: usize) -> Vec<T> {
        let taps = kernel_size * 2 + 1;

        // Precompute Hann window
        let taps_minus_1 = T::from_usize(taps - 1);
        let mut window = vec![0.0.into(); taps];
        for i in 0..taps {
            let x = T::from_usize(i) / taps_minus_1;
            window[i] = if TypeId::of::<T>() == TypeId::of::<f64>() {
                let xf = x.to_f64();
                T::from_f64(0.5 - 0.5 * (2.0 * std::f64::consts::PI * xf).cos())
            } else {
                let xf = x.to_f32();
                T::from_f32(0.5 - 0.5 * (2.0 * std::f32::consts::PI * xf).cos())
            };
        }

        return window;
    }
}

impl<T: FloatType + 'static> ResamplerImpl<T> for Resampler<T> {
    fn new(quality: ResamplerQuality, layout: AudioChannelLayout) -> Resampler<T> {
        let kernel_size = quality.to_converter_type() as usize;
        let window = Resampler::alloc_window(kernel_size);

        return Resampler {
            ch_layout: layout,
            _ph: PhantomData::default(),
            kernel_size,
            window,
            quality
        };
    }

    fn process(&mut self, input: &[T], output: &mut [T], src_ratio: f64) -> usize {
        unsafe {
            if src_ratio == 1.0 {
                std::ptr::copy_nonoverlapping(input.as_ptr(), output.as_mut_ptr(), input.len());
                return input.len();
            }

            let channels = self.ch_layout as usize;
            if input.is_empty() {
                return 0;
            }

            let src_ratio = 1.0 / src_ratio.clamp(MIN_RESAMPLE_RATIO, MAX_RESAMPLE_RATIO);

            let frames_in = input.len() / channels;
            let frames_out = ((frames_in as f64) * src_ratio).ceil() as usize;

            let cutoff: T = T::from_f64(0.95);
            let src_ratio: T = T::from_f64(src_ratio);

            #[inline]
            fn sinc<T: FloatType + 'static>(x: T) -> T {
                if x.abs() < T::from_f64(1e-6) {
                    return 1.0.into();
                }
                if TypeId::of::<T>() == TypeId::of::<f64>() {
                    let xf = x.to_f64();
                    T::from_f64((std::f64::consts::PI * xf).sin() / (std::f64::consts::PI * xf))
                } else {
                    let xf = x.to_f32();
                    T::from_f32((std::f32::consts::PI * xf).sin() / (std::f32::consts::PI * xf))
                }
            }

            for ch in 0..channels {
                for n in 0..frames_out {
                    let src_pos = T::from_usize(n) / src_ratio; // fractional input position
                    let center = src_pos.floor().to_isize();

                    let mut acc: T = 0.0.into();
                    let mut norm: T = 0.0.into();

                    for k in -(self.kernel_size as isize)..=(self.kernel_size as isize) {
                        let idx = center + k;
                        if idx < 0 || idx >= frames_in as isize {
                            continue;
                        }

                        let idx_us = idx as usize;
                        let frac = src_pos - T::from_usize(idx_us);

                        let tap_index = (k + self.kernel_size as isize) as usize;

                        // bandlimited sinc
                        let sinc_val = cutoff * sinc(frac * cutoff);
                        let tap = sinc_val * *self.window.as_ptr().add(tap_index);

                        let sample = *input.as_ptr().add(idx_us * channels + ch);
                        acc += sample * tap;
                        norm += tap;
                    }

                    let out_sample = if norm.abs() > T::from_f64(1e-12) {
                        acc / norm
                    } else {
                        0.0.into()
                    };

                    output[n * channels + ch] = out_sample;
                }
            }

            let out = output.as_ptr() as *const f32;

            return frames_out;
        }
    }

    /// Set other ResamplerQuality.
    /// Reallocates SRC_STATE
    fn set_quality(&mut self, quality: ResamplerQuality) {
        if self.quality == quality {
            return;
        }
        let new_ksize = quality.to_converter_type() as usize;

        self.window = Resampler::alloc_window(new_ksize);
        self.kernel_size = new_ksize;
        self.quality = quality;
    }

    /// ResamplerQuality getter.
    fn get_quality(&self) -> ResamplerQuality {
        return self.quality
    }

    fn get_channel_layout(&self) -> AudioChannelLayout {
        return self.ch_layout;
    }

    fn set_channel_layout(&mut self, layout: AudioChannelLayout) {
        self.ch_layout = layout;
    }

    fn reset(&mut self) {}
}

// pub fn resample_windowed_sinc(input: &[T], output: &mut [T], ratio: f64) {
//     if input.is_empty() {
//         return Vec::new();
//     }
//     let kernel_size = 32;
//     let cutoff = 0.95f32;
//     let out_len = (input.len() as f32 * ratio).ceil() as usize;

//     let taps = kernel_size * 2 + 1;
//     let mut output = Vec::with_capacity(out_len);

//     // Precompute Hann window
//     let mut window = vec![0.0f32; taps];
//     for i in 0..taps {
//         let x = i as f32 / (taps - 1) as f32;
//         window[i] = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos();
//     }

//     // Sinc helper
//     #[inline]
//     fn sinc(x: f32) -> f32 {
//         if x.abs() < 1e-6 {
//             1.0
//         } else {
//             (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x)
//         }
//     }

//     for n in 0..out_len {
//         let src_pos = n as f32 / ratio;
//         let center = src_pos.floor() as isize;

//         let mut acc = 0.0f32;
//         let mut norm = 0.0f32;

//         for k in -(kernel_size as isize)..=(kernel_size as isize) {
//             let idx = center + k;
//             if idx < 0 || idx >= input.len() as isize {
//                 continue;
//             }

//             let frac = src_pos - (idx as f32);
//             let tap_index = (k + kernel_size as isize) as usize;

//             // Bandlimited sinc with cutoff
//             let sinc_val = cutoff * sinc(frac * cutoff);
//             let w = window[tap_index];
//             let tap = sinc_val * w;

//             acc += input[idx as usize] * tap;
//             norm += tap;
//         }

//         output.push(if norm.abs() > 1e-12 { acc / norm } else { 0.0 });
//     }

//     output
// }
