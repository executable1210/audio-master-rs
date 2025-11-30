use biquad::{Coefficients, DirectForm1, Q_BUTTERWORTH_F32, ToHertz, Type};
use num_traits::Float;

use crate::{
    audio_buffer::{AudioBuffer, AudioBufferImpl, AudioChannelLayout},
    consts::{MAX_FREQ_CUT, MIN_FREQ_CUT},
    float_type::FloatType,
};

use super::audio_math::{AudioFilter, AudioFilterImpl, AudioFilterPrivateImpl};

impl<T: FloatType + Float> AudioFilterImpl<T> for AudioFilter<T> {
    fn new(
        low_cut: f32,
        high_cut: f32,
        sample_rate: f32,
        channel_layout: AudioChannelLayout,
    ) -> Self {
        assert!(
            sample_rate > 0.0,
            "Sample rate must be positive, got {}",
            sample_rate
        );
        assert!(
            low_cut < high_cut,
            "Low cutoff ({:.2} Hz) must be less than high cutoff ({:.2} Hz)",
            low_cut,
            high_cut
        );

        let low_cut = low_cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT);
        let high_cut = high_cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT);

        let highpass = Coefficients::<T>::from_params(
            Type::HighPass,
            sample_rate.hz(),
            low_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        let lowpass = Coefficients::<T>::from_params(
            Type::LowPass,
            sample_rate.hz(),
            high_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        let mut lowpasses: Vec<DirectForm1<T>> = Vec::new();
        lowpasses.resize_with(channel_layout as usize, || DirectForm1::<T>::new(lowpass));

        let mut highpasses: Vec<DirectForm1<T>> = Vec::new();
        highpasses.resize_with(channel_layout as usize, || DirectForm1::<T>::new(highpass));

        return Self {
            lowpass: lowpasses,
            highpass: highpasses,
            low_cut,
            high_cut,
            sample_rate,
            channel_layout,
        };
    }

    fn set_channel_layout(&mut self, channel_layout: AudioChannelLayout) {
        if self.channel_layout != channel_layout {
            *self = AudioFilter::new(
                self.low_cut,
                self.high_cut,
                self.sample_rate,
                channel_layout,
            );
        }
    }

    fn process(&mut self, input: &mut AudioBuffer<T>) {
        assert!(
            input.channel_layout() == self.channel_layout,
            "Input channel layout ({:?}) does not match filter channel layout ({:?})",
            input.channel_layout(),
            self.channel_layout
        );

        for ch in 0..self.channel_layout as usize {
            let channel_data = input.get_channel_mut(ch);
            self.process_channel(channel_data, ch);
        }
    }

    fn set_low_cut(&mut self, low_cut: f32) {
        let low_cut = low_cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT);
        assert!(
            low_cut < self.high_cut,
            "Low cutoff ({:.2} Hz) must be less than high cutoff ({:.2} Hz)",
            low_cut,
            self.high_cut
        );

        let highpass = Coefficients::<T>::from_params(
            Type::HighPass,
            self.sample_rate.hz(),
            low_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        for hp in self.highpass.iter_mut() {
            *hp = DirectForm1::<T>::new(highpass);
        }
    }

    fn set_high_cut(&mut self, high_cut: f32) {
        let high_cut = high_cut.clamp(MIN_FREQ_CUT, MAX_FREQ_CUT);
        assert!(
            high_cut > self.low_cut,
            "High cutoff ({:.2} Hz) must be greater than low cutoff ({:.2} Hz)",
            high_cut,
            self.low_cut
        );

        let lowpass = Coefficients::<T>::from_params(
            Type::LowPass,
            self.sample_rate.hz(),
            high_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        for lp in self.lowpass.iter_mut() {
            *lp = DirectForm1::<T>::new(lowpass);
        }
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        if self.sample_rate == sample_rate {
            return;
        }

        assert!(
            sample_rate > 0.0,
            "Sample rate must be positive, got {}",
            sample_rate
        );
        self.sample_rate = sample_rate;

        let highpass = Coefficients::<T>::from_params(
            Type::HighPass,
            sample_rate.hz(),
            self.low_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        let lowpass = Coefficients::<T>::from_params(
            Type::LowPass,
            sample_rate.hz(),
            self.high_cut.hz(),
            Q_BUTTERWORTH_F32.into(),
        )
        .unwrap();

        for lp in self.lowpass.iter_mut() {
            *lp = DirectForm1::<T>::new(lowpass);
        }

        for hp in self.highpass.iter_mut() {
            *hp = DirectForm1::<T>::new(highpass);
        }
    }

    #[inline]
    fn get_high_cut(&self) -> f32 {
        return self.high_cut;
    }

    #[inline]
    fn get_low_cut(&self) -> f32 {
        return self.low_cut;
    }

    #[inline]
    fn get_sample_rate(&self) -> f32 {
        return self.sample_rate;
    }
}
