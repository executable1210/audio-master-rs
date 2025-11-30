use biquad::Biquad;
use num_traits::Float;

use crate::float_type::FloatType;

use super::audio_math::{AudioFilter, AudioFilterPrivateImpl, AudioMath, AudioMathPrivateImpl};

impl<T: FloatType> AudioMathPrivateImpl<T> for AudioMath {
    #[inline]
    fn db_to_gain(db: T) -> T {
        let a: T = 10.0.into();
        return a.powf(db / 20.0.into());
    }

    #[inline]
    fn gain_to_db(gain: T) -> T {
        let a: T = 20.0.into();
        return a * gain.max(0.0.into()).log10();
    }
}

impl<T: FloatType + Float> AudioFilterPrivateImpl<T> for AudioFilter<T> {
    #[inline]
    fn process_channel(&mut self, input: &mut [T], channel: usize) {
        for sample in input.iter_mut() {
            let y = self.highpass[channel].run(*sample);
            let y = self.lowpass[channel].run(y);

            *sample = y;
        }
    }
}
