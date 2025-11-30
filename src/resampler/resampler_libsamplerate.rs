use std::{
    any::TypeId,
    ffi::{CStr, c_double, c_int, c_long},
    marker::PhantomData,
    ptr,
};

use crate::{
    audio_buffer::AudioChannelLayout,
    consts::{MAX_RESAMPLE_RATIO, MIN_RESAMPLE_RATIO},
    ffi::libsamplerate::{
        SRC_CONVERTER_TYPE, SRC_DATA, SRC_STATE, src_delete, src_new, src_process, src_reset,
        src_strerror,
    },
    float_type::FloatType,
};

use super::base::{Resampler, ResamplerImpl, ResamplerQuality};

trait ResamplerPrivateImpl<T: FloatType>: Sized {
    /// Panics if a process error occured.
    fn _process(
        &mut self,
        input: &[T],
        output: &mut [T],
        src_ratio: f64,
        is_last_chunk: bool,
    ) -> usize;

    /// Drop implementation.
    fn drop_impl(&mut self);

    fn free_state(&mut self);

    /// Panic if a C side error occured.
    fn create_state(layout: AudioChannelLayout, quality: ResamplerQuality) -> *mut SRC_STATE;
}

impl<T: FloatType + 'static> ResamplerImpl<T> for Resampler<T> {
    fn new(quality: ResamplerQuality, layout: AudioChannelLayout) -> Resampler<T> {
        return Resampler {
            state: Resampler::<T>::create_state(layout, quality),
            ch_layout: layout,
            quality,
            _phantom: PhantomData::default(),
        };
    }

    fn process(&mut self, input: &[T], output: &mut [T], src_ratio: f64) -> usize {
        if src_ratio == 1.0 {
            unsafe {
                ptr::copy(input.as_ptr(), output.as_mut_ptr(), input.len());
            }
            return input.len() as usize / self.ch_layout as usize;
        }

        return self._process(input, output, 1.0 / src_ratio, false);
    }

    fn set_quality(&mut self, quality: ResamplerQuality) {
        self.free_state();
        self.state = Resampler::<T>::create_state(self.ch_layout, quality);
        self.quality = quality;
    }

    #[inline]
    fn get_quality(&self) -> ResamplerQuality {
        return self.quality;
    }

    #[inline]
    fn get_channel_layout(&self) -> AudioChannelLayout {
        return self.ch_layout;
    }

    fn set_channel_layout(&mut self, layout: AudioChannelLayout) {
        if self.ch_layout == layout {
            return;
        }

        *self = Resampler::new(self.quality, layout);
    }

    fn reset(&mut self) {
        let result = unsafe { src_reset(self.state) };
        if result != 0 {
            let error_msg = unsafe { CStr::from_ptr(src_strerror(result)) }
                .to_string_lossy()
                .into_owned();

            panic!("Reset error: {}", error_msg);
        }
    }
}

impl<T: FloatType + 'static> ResamplerPrivateImpl<T> for Resampler<T> {
    fn _process(
        &mut self,
        input: &[T],
        output: &mut [T],
        src_ratio: f64,
        is_last_chunk: bool,
    ) -> usize {
        unsafe {
            let src_ratio = src_ratio.clamp(MIN_RESAMPLE_RATIO, MAX_RESAMPLE_RATIO);

            let input_frames = (input.len() / self.ch_layout as usize) as c_long;

            let mut _input: *const f32 = std::ptr::null();
            let mut _output: *mut f32 = std::ptr::null_mut();

            let mut inputf64_vec: Vec<f32> = Vec::default();
            let mut outputf64_vec: Vec<f32> = Vec::default();

            let mut is_f64 = false;

            // Since libsamplerate doesn't support f64 sample process
            // so we need to convert input into f32.
            if TypeId::of::<T>() == TypeId::of::<f64>() {
                is_f64 = true;

                inputf64_vec = input.iter().map(|x| x.to_f32()).collect();
                outputf64_vec = Vec::with_capacity(output.len());
                outputf64_vec.set_len(output.len());
                _input = inputf64_vec.as_ptr();
                _output = outputf64_vec.as_mut_ptr();
            } else {
                _input =
                    std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len()).as_ptr();

                _output =
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
                        .as_mut_ptr();
            }

            let mut data = SRC_DATA {
                data_in: _input,
                data_out: _output,
                input_frames,
                output_frames: ((*output).len() / self.ch_layout as usize) as c_long,
                input_frames_used: 0,
                output_frames_gen: 0,
                end_of_input: if is_last_chunk { 1 } else { 0 },
                src_ratio: src_ratio as c_double,
            };

            let result = unsafe { src_process(self.state, &mut data) };
            if result != 0 {
                let error_msg = unsafe { CStr::from_ptr(src_strerror(result)) }
                    .to_string_lossy()
                    .into_owned();

                panic!("Resampler process error: {}", error_msg);
            }

            let output_len = (data.output_frames_gen as usize);

            if TypeId::of::<T>() == TypeId::of::<f64>() {
                let output =
                    std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f64, input.len())
                        .as_mut_ptr();
                for i in 0..output_len as usize {
                    *output.add(i) = *_output.add(i) as f64;
                }
            }

            return output_len;
        }
    }

    fn drop_impl(&mut self) {
        self.free_state();
    }

    fn free_state(&mut self) {
        unsafe {
            src_delete(self.state);
        }
    }

    fn create_state(layout: AudioChannelLayout, quality: ResamplerQuality) -> *mut SRC_STATE {
        unsafe {
            let mut error: c_int = 0;
            let state = src_new(quality.to_converter_type(), layout as c_int, &mut error);

            if state.is_null() {
                let error_msg = CStr::from_ptr(src_strerror(error))
                    .to_string_lossy()
                    .into_owned();
                panic!("Failed to create converter: {}", error_msg);
            }

            return state;
        }
    }
}

impl ResamplerQuality {
    fn to_converter_type(&self) -> i32 {
        match self {
            ResamplerQuality::Linear => SRC_CONVERTER_TYPE::SRC_LINEAR as i32,
            ResamplerQuality::SincLow => SRC_CONVERTER_TYPE::SRC_SINC_FASTEST as i32,
            ResamplerQuality::SincMedium => SRC_CONVERTER_TYPE::SRC_SINC_MEDIUM_QUALITY as i32,
            ResamplerQuality::SincHigh => SRC_CONVERTER_TYPE::SRC_SINC_BEST_QUALITY as i32,
        }
    }
}
