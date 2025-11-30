use crate::float_type::FloatType;

use super::audio_math::{AudioMath, AudioMathImpl, AudioMathPrivateImpl};

impl<T: FloatType + Clone> AudioMathImpl<T> for AudioMath {
    // fn mix_signals_inplace_normalized(a: &mut [T], b: &[T], gain: f32) {
    //     for (x, y) in a.iter_mut().zip(b.iter()) {
    //         let mixed = *x + T::from(gain) * *y;
    //         *x = mixed.clamp(T::from(-1.0), T::from(1.0));
    //     }
    // }

    fn mix_signals_inplace(a: &mut [T], b: &[T], gain: f32) {
        for (s1, s2) in a.iter_mut().zip(b.iter()) {
            *s1 = (*s1 + *s2 * T::from(gain)) * T::from(0.5);
        }
    }

    fn mix_signals(a: &mut [T], b: &[T]) {
        AudioMath::mix_signals_inplace(a, b, 1.0);
    }

    fn normalize_rms(buffer: &mut [T], rms_factor: T) {
        assert!(rms_factor > 0.0.into(), "Target RMS must be positive");

        let sum_squares: T = buffer.iter().map(|&s| s * s).sum();
        let rms = (sum_squares / T::from_usize(buffer.len())).sqrt();

        if rms == 0.0.into() {
            return;
        }

        let factor = rms_factor / rms;
        for sample in buffer.iter_mut() {
            *sample = *sample * factor;
        }
    }

    fn normalize(buffer: &mut [T], factor: T) {
        assert!(factor > 0.0.into(), "Target peak must be positive");

        let mut current_peak: T = 0.0.into();

        // Find the maximum absolute value in the buffer
        for sample in buffer.iter_mut() {
            if sample.is_nan() {
                *sample = 0.0.into();
            }
            current_peak = current_peak.max(sample.abs());
        }

        if current_peak == 0.0.into() {
            return;
        }

        let normalization_factor = factor / current_peak;

        for sample in buffer.iter_mut() {
            *sample = *sample * normalization_factor;
        }
    }

    fn scale_volume_db(audio: &mut [T], db: T) {
        let gain = Self::db_to_gain(db);
        for sample in audio.iter_mut() {
            *sample = *sample * gain;
        }
    }

    fn scale_volume_db_clipped(audio: &mut [T], db: T) {
        let gain = Self::db_to_gain(db);
        for sample in audio.iter_mut() {
            *sample = (*sample * gain).clamp((-1.0).into(), 1.0.into());
        }
    }

    fn normalize_to_max_db(audio: &mut [T], target_max_db: T) {
        // Find current peak amplitude
        let current_peak = audio
            .iter()
            .fold(0.0.into(), |max, &sample| sample.abs().max(max));

        if current_peak > 0.0.into() {
            let target_amplitude = Self::db_to_gain(target_max_db);
            let current_amplitude_db = Self::gain_to_db(current_peak);
            let required_gain_db = target_max_db - current_amplitude_db;
            let gain = Self::db_to_gain(required_gain_db);

            for sample in audio.iter_mut() {
                *sample = *sample * gain;
            }
        }
    }

    fn normalize_with_headroom(audio: &mut [T], target_max_db: T, headroom_db: T) {
        Self::normalize_to_max_db(audio, target_max_db + headroom_db);
    }
}
