use crate::audio_math::{AudioMath, AudioMathImpl};
use crate::consts::MAX_AUDIO_CHANNELS;
use crate::float_type::FloatType;

// const INV_SQRT2: f64 = 0.7071067811865476;
const INV_SQRT2: f32 = 0.7071067811865476;

/// Represents common audio channel layouts, where the enum variant's repr(usize) corresponds
/// to the number of channels in that layout.
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioChannelLayout {
    /// Single channel audio (monaural).
    Mono = 1,
    /// Two-channel audio (left and right).
    Stereo = 2,
    /// Three-channel surround sound (left, right, center).
    Surround30 = 3,
    /// Four-channel surround sound (left, right, center, rear).
    Surround40 = 4,
    /// Six-channel surround sound (5.1: left, right, center, LFE, left surround, right surround).
    Surround51 = 6,
    /// Eight-channel surround sound (7.1: adds rear left and rear right).
    Surround71 = 8,
}

impl Default for AudioChannelLayout {
    fn default() -> Self {
        Self::Stereo
    }
}

/// Represents individual audio channels in a multi-channel setup, with repr(usize) for indexing.
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub enum AudioChannel {
    /// Left channel.
    Left = 0,
    /// Right channel.
    Right = 1,
    /// Center channel.
    Center = 2,
    /// Low-Frequency Effects (subwoofer) channel.
    LFE = 3,
    /// Left surround channel.
    LeftSurround = 4,
    /// Right surround channel.
    RightSurround = 5,
    /// Rear left channel (for 7.1 layouts).
    RearLeft = 6,
    /// Rear right channel (for 7.1 layouts).
    RearRight = 7,
}

pub trait AudioBufferMathImpl<T: FloatType>: Sized {
    /// This adjusts the audio levels to a target RMS value.
    fn normalize_rms(&mut self, rms_factor: T);

    /// Useful for gain adjustments.
    fn normalize(&mut self, factor: T);
}

/// Trait for interleaved audio buffer operations, where samples from all channels are stored
/// in a single contiguous buffer (e.g., LRLRLR... for stereo).
pub trait AudioBufferInterleavedImpl<T: FloatType>: Sized {
    /// Creates a new interleaved audio buffer with the specified length (per channel) and channel layout.
    /// The total buffer length will be `length * layout as usize`.
    fn new(length: usize, layout: AudioChannelLayout) -> AudioBufferInterleaved<T>;

    /// Resets the buffer by filling it with zeros.
    fn reset(&mut self);

    /// Returns the current channel layout of the buffer.
    fn get_channel_layout(&self) -> AudioChannelLayout;

    /// Sets a new channel layout, potentially reallocating the buffer if the layout changes.
    fn set_channel_layout(&mut self, layout: AudioChannelLayout);

    /// Returns an immutable reference to the entire interleaved buffer.
    fn get_buffer(&self) -> &[T];

    /// Returns a mutable reference to the entire interleaved buffer.
    fn get_buffer_mut(&mut self) -> &mut [T];

    /// Returns the length per channel (planar length).
    fn len_planar(&self) -> usize;

    /// Returns the total interleaved length (channels * planar length).
    fn len_interleaved(&self) -> usize;

    /// Resizes the buffer to a new planar length, adjusting the total size accordingly.
    fn resize(&mut self, new_len: usize);

    /// Copies the interleaved data into a planar AudioBuffer (separate buffers per channel).
    fn copy_to_planar(&self, output: &mut AudioBuffer<T>);
}

/// Trait for planar audio buffer operations, where each channel has its own separate buffer.
pub trait AudioBufferImpl<T: FloatType>: Sized {
    /// Creates a new planar audio buffer with the specified length (per channel) and channel layout.
    fn new(len: usize, layout: AudioChannelLayout) -> AudioBuffer<T>;

    /// Creates a new planar audio buffer with default length (0) and the specified layout.
    fn new_with_layout(layout: AudioChannelLayout) -> AudioBuffer<T>;

    /// Returns an immutable slice for the specified channel.
    /// Panics if the channel index exceeds the layout's channel count.
    fn get_channel(&self, channel: usize) -> &[T];

    /// Returns a mutable slice for the specified channel.
    /// Panics if the channel index exceeds the layout's channel count.
    fn get_channel_mut(&mut self, channel: usize) -> &mut [T];

    /// Returns an immutable slice for the specified channel without bounds checking.
    unsafe fn get_channel_unchecked(&self, channel: usize) -> &[T];

    /// Returns a mutable slice for the specified channel without bounds checking.
    unsafe fn get_channel_unchecked_mut(&mut self, channel: usize) -> &mut [T];

    /// Resizes all channel buffers to the new length, filling new space with zeros.
    fn resize(&mut self, new_len: usize);

    /// Copies all channels' data into an interleaved destination slice.
    /// Panics if the destination length does not match `self.len() * self.channel_layout() as usize`.
    fn copy_to_interleaved(&self, dst: &mut [T]);

    /// Mixes another AudioBuffer into this one without applying gain.
    /// Panics if lengths or layouts do not match.
    fn mix_with(&mut self, other: &AudioBuffer<T>);

    /// Mixes another AudioBuffer into this one with the specified gain applied to the other buffer.
    /// Panics if lengths or layouts do not match.
    fn mix_with_gain(&mut self, other: &AudioBuffer<T>, gain: f32);

    /// Copies data from another AudioBuffer.
    /// Panics if lengths or layouts do not match.
    fn copy_from(&mut self, other: &AudioBuffer<T>);

    /// Copies data from an interleaved buffer.
    /// Panics if the interleaved length does not match the expected size.
    fn copy_from_interleaved(&mut self, buffer: &[T]);

    /// Copies a range from an interleaved input buffer, starting at an offset.
    /// - `start` and `end`: Range in the input buffer.
    /// - `offset`: Starting channel offset in the planar buffer.
    /// Panics if `start > end` or `end >= input.len()`.
    fn copy_from_interleaved_range(&mut self, input: &[T], start: usize, end: usize, offset: usize);

    /// Returns the current channel layout.
    fn channel_layout(&self) -> AudioChannelLayout;

    /// Sets a new channel layout, converting the audio signal into new layout.
    fn set_channel_layout(&mut self, new_layout: AudioChannelLayout);

    /// Resets all buffers by filling them with zeros.
    fn reset(&mut self);

    /// Returns the length per channel (planar length).
    fn len(&self) -> usize;

    /// Returns the total interleaved length (channels * planar length).
    fn len_interleaved(&self) -> usize;

    /// Converts this buffer to a cloned AudioBuffer with f32 samples.
    fn to_f32(&self) -> AudioBuffer<f32>;

    /// Converts this buffer to a cloned AudioBuffer with f64 samples.
    fn to_f64(&self) -> AudioBuffer<f64>;
}

#[derive(Debug, Clone)]
pub struct AudioBuffer<T: FloatType> {
    buffers: [Vec<T>; MAX_AUDIO_CHANNELS],
    /// Total interleaved length (layout * len())
    len_interleaved: usize, 
    layout: AudioChannelLayout,
}

#[derive(Debug, Clone)]
pub struct AudioBufferInterleaved<T: FloatType> {
    buffer: Vec<T>,
    /// Length per channel.
    len_planar: usize,
    layout: AudioChannelLayout,
}

impl<T: FloatType> AudioBufferInterleavedImpl<T> for AudioBufferInterleaved<T> {
    fn new(len: usize, layout: AudioChannelLayout) -> AudioBufferInterleaved<T> {
        unsafe {
            let len_interleaved = len * layout as usize;
            let mut buffer: Vec<T> = Vec::with_capacity(len_interleaved);
            buffer.set_len(len_interleaved);
            buffer.fill(0.0.into());

            let len_planar = len;

            return AudioBufferInterleaved {
                buffer,
                layout,
                len_planar,
            };
        }
    }

    fn reset(&mut self) {
        self.buffer.fill(0.0.into());
    }

    fn get_channel_layout(&self) -> AudioChannelLayout {
        return self.layout;
    }

    fn set_channel_layout(&mut self, layout: AudioChannelLayout) {
        if self.layout != layout {
            *self = AudioBufferInterleaved::new(self.len_planar(), layout);
        }
    }

    fn get_buffer(&self) -> &[T] {
        return self.buffer.as_slice();
    }

    fn get_buffer_mut(&mut self) -> &mut [T] {
        return self.buffer.as_mut_slice();
    }

    fn len_planar(&self) -> usize {
        return self.len_planar;
    }

    fn len_interleaved(&self) -> usize {
        return self.buffer.len();
    }

    fn resize(&mut self, new_len: usize) {
        self.len_planar = new_len;
        self.buffer
            .resize(new_len * self.layout as usize, 0.0.into());
    }

    fn copy_to_planar(&self, output: &mut AudioBuffer<T>) {
        output.copy_from_interleaved(&self.buffer);
    }
}

impl<T: FloatType> AudioBufferMathImpl<T> for AudioBufferInterleaved<T> {
    fn normalize_rms(&mut self, rms_factor: T) {
        AudioMath::normalize_rms(&mut self.buffer.as_mut_slice(), rms_factor);
    }

    fn normalize(&mut self, factor: T) {
        AudioMath::normalize(&mut self.buffer.as_mut_slice(), factor);
    }
}

impl<T: FloatType> AudioBufferMathImpl<T> for AudioBuffer<T> {
    fn normalize_rms(&mut self, rms_factor: T) {
        for buffer in self.buffers.iter_mut() {
            AudioMath::normalize_rms(buffer.as_mut_slice(), rms_factor);
        }
    }

    fn normalize(&mut self, factor: T) {
        for buffer in self.buffers.iter_mut() {
            AudioMath::normalize(buffer.as_mut_slice(), factor);
        }
    }
}

impl<T: FloatType> AudioBufferImpl<T> for AudioBuffer<T> {
    fn new(len: usize, layout: AudioChannelLayout) -> AudioBuffer<T> {
        let mut buffers: [Vec<T>; 8] = std::array::from_fn(|_| Vec::with_capacity(len));

        for bf in buffers.iter_mut() {
            unsafe {
                bf.set_len(len);
            }
        }

        let len_interleaved = len * layout as usize;

        return AudioBuffer {
            buffers,
            layout,
            len_interleaved,
        };
    }

    fn new_with_layout(layout: AudioChannelLayout) -> AudioBuffer<T> {
        let mut buffers: [Vec<T>; 8] = std::array::from_fn(|_| Vec::default());

        for bf in buffers.iter_mut() {
            unsafe {
                bf.set_len(0);
            }
        }

        let len_interleaved = 0usize;

        return AudioBuffer {
            buffers,
            layout,
            len_interleaved,
        };
    }

    #[inline]
    fn get_channel(&self, channel: usize) -> &[T] {
        assert!(channel < self.layout as usize);

        unsafe {
            return self.buffers.get_unchecked(channel);
        }
    }

    #[inline]
    fn get_channel_mut(&mut self, channel: usize) -> &mut [T] {
        assert!(channel < self.layout as usize);

        unsafe {
            return self.buffers.get_unchecked_mut(channel);
        }
    }

    #[inline]
    unsafe fn get_channel_unchecked(&self, channel: usize) -> &[T] {
        unsafe {
            return self.buffers.get_unchecked(channel);
        }
    }

    #[inline]
    unsafe fn get_channel_unchecked_mut(&mut self, channel: usize) -> &mut [T] {
        unsafe {
            return self.buffers.get_unchecked_mut(channel);
        }
    }

    fn resize(&mut self, new_len: usize) {
        for bf in self.buffers.iter_mut() {
            bf.resize(new_len, 0.0.into());
        }

        self.len_interleaved = new_len * self.layout as usize;
    }

    fn copy_to_interleaved(&self, dst: &mut [T]) {
        assert!(dst.len() == self.len() * self.layout as usize);

        let dst = dst.as_mut_ptr();
        let buffers = self.get_buffers();

        unsafe {
            for i in 0..self.len() {
                for channel in 0..self.layout as usize {
                    *dst.add(i * self.layout as usize + channel) =
                        *(*buffers.add(channel)).as_mut_ptr().add(i);
                }
            }
        }
    }

    fn mix_with(&mut self, other: &AudioBuffer<T>) {
        assert!(other.len() == self.len(), "Buffers length don't match");
        assert!(
            other.channel_layout() == self.channel_layout(),
            "Buffers channel layout don't match",
        );

        unsafe {
            let buffers = self.get_buffers();
            let other_buffers = other.get_buffers();

            for ch in 0..self.channel_layout() as usize {
                let left = buffers.add(ch);
                let right = other_buffers.add(ch);

                AudioMath::mix_signals_inplace((*left).as_mut_slice(), (*right).as_slice(), 1.0);
            }
        }
    }

    fn mix_with_gain(&mut self, other: &AudioBuffer<T>, gain: f32) {
        assert!(other.len() == self.len(), "Buffer lengths do not match");
        assert!(
            other.channel_layout() == self.channel_layout(),
            "Buffer channel layouts do not match",
        );

        unsafe {
            let buffers = self.get_buffers();
            let other_buffers = other.get_buffers();

            for ch in 0..self.channel_layout() as usize {
                let left = buffers.add(ch);
                let right = other_buffers.add(ch);

                AudioMath::mix_signals_inplace((*left).as_mut_slice(), (*right).as_slice(), gain);
            }
        }
    }

    fn copy_from(&mut self, other: &AudioBuffer<T>) {
        assert!(other.len() == self.len(), "Buffer lengths do not match");
        assert!(
            other.channel_layout() == self.channel_layout(),
            "Buffer channel layouts do not match",
        );

        unsafe {
            let buffers = self.get_buffers();
            let other_buffers = other.get_buffers();

            for i in 0..self.layout as usize {
                std::ptr::copy_nonoverlapping(
                    (*other_buffers.add(i)).as_ptr(),
                    (*buffers.add(i)).as_mut_ptr(),
                    self.len(),
                );
            }
        }
    }

    fn copy_from_interleaved(&mut self, buffer: &[T]) {
        assert!(buffer.len() * self.layout as usize == self.len());

        let channels = self.layout as usize;

        unsafe {
            for i in 0..self.len() {
                for ch in 0..channels {
                    *self.get_channel_mut(ch).as_mut_ptr().add(i) =
                        *buffer.as_ptr().add(i * channels + ch);
                }
            }
        }
    }

    fn copy_from_interleaved_range(
        &mut self,
        input: &[T],
        start: usize,
        end: usize,
        offset: usize,
    ) {
        assert!(start < end || end <= input.len());

        unsafe {
            let num_channels = self.channel_layout() as usize;
            let num_samples = end - start;

            for channel in 0..num_channels {
                let dst = self.get_channel_unchecked_mut(channel).as_mut_ptr();
                let input = input.as_ptr();
                for i in 0..num_samples {
                    *dst.add(offset + i) = *input.add((start + i) * num_channels + channel);
                }
            }
        }
    }

    #[inline]
    fn channel_layout(&self) -> AudioChannelLayout {
        return self.layout;
    }

    fn set_channel_layout(&mut self, new_layout: AudioChannelLayout) {
        if self.layout == new_layout {
            return;
        }

        self.len_interleaved = self.len() * new_layout as usize;

        match new_layout {
            AudioChannelLayout::Mono => {
                self.into_mono();
            }
            AudioChannelLayout::Stereo => {
                self.into_stereo();
            }
            AudioChannelLayout::Surround30 => {
                self.into_surround30();
            }
            AudioChannelLayout::Surround40 => {
                self.into_surround40();
            }
            AudioChannelLayout::Surround51 => {
                self.into_surround51();
            }
            AudioChannelLayout::Surround71 => {
                self.into_surround71();
            }
        }
    }

    fn reset(&mut self) {
        for ch in self.buffers.iter_mut() {
            ch.fill(T::from(0.0)); // ismatched types expected type parameter `T` found type `{float}`
        }
    }

    #[inline]
    fn len(&self) -> usize {
        unsafe {
            return self.buffers.get_unchecked(0).len();
        }
    }

    #[inline]
    fn len_interleaved(&self) -> usize {
        return self.len_interleaved;
    }

    #[inline]
    fn to_f32(&self) -> AudioBuffer<f32> {
        unsafe {
            let mut new = AudioBuffer::<f32>::new(self.len(), self.layout);

            for i in 0..self.layout as usize {
                let new_buff = new.get_channel_unchecked_mut(i).as_mut_ptr();
                for (y, s) in self.buffers.get_unchecked(i).iter().enumerate() {
                    *new_buff.add(y) = s.to_f32();
                }
            }

            return new;
        }
    }

    #[inline]
    fn to_f64(&self) -> AudioBuffer<f64> {
        unsafe {
            let mut new = AudioBuffer::<f64>::new(self.len(), self.layout);

            for i in 0..self.layout as usize {
                let new_buff = new.get_channel_unchecked_mut(i).as_mut_ptr();
                for (y, s) in self.buffers.get_unchecked(i).iter().enumerate() {
                    *new_buff.add(y) = s.to_f64();
                }
            }

            return new;
        }
    }
}

impl<T: FloatType> AudioBuffer<T> {
    #[inline]
    pub(super) fn get_buffers(&self) -> *mut Vec<T> {
        let ptr = self.buffers.as_ptr();

        return ptr as *mut Vec<T>;
    }
}

impl<T: FloatType> AudioBuffer<T> {
    fn into_mono(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {}
            AudioChannelLayout::Stereo => {
                unsafe {
                    // L_mono = T::from(0.5) * L + T::from(0.5) * R
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    for i in 0..len {
                        *left.add(i) = T::from(0.5) * *left.add(i) + T::from(0.5) * *right.add(i);
                    }
                    self.layout = AudioChannelLayout::Mono;
                }
            }
            AudioChannelLayout::Surround30 => {
                unsafe {
                    // L_mono = T::from(0.5) * L + T::from(0.5) * R + 0.707 * C
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    for i in 0..len {
                        *left.add(i) = T::from(0.5) * *left.add(i)
                            + T::from(0.5) * *right.add(i)
                            + T::from(INV_SQRT2) * *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Mono;
                }
            }
            AudioChannelLayout::Surround40 => {
                unsafe {
                    // L_mono = T::from(0.5) * L + T::from(0.5) * R + T::from(0.5) * Ls + T::from(0.5) * Rs
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();
                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();

                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    for i in 0..len {
                        *left.add(i) = T::from(0.5) * *left.add(i)
                            + T::from(0.5) * *right.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Mono;
                }
            }
            AudioChannelLayout::Surround51 => {
                unsafe {
                    // L_mono = T::from(0.5) * L + T::from(0.5) * R + 0.707 * C + T::from(0.5) * Ls + T::from(0.5) * Rs + T::from(0.5) * LFE
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    for i in 0..len {
                        *left.add(i) = T::from(0.5) * *left.add(i)
                            + T::from(0.5) * *right.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *lfe.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Mono;
                }
            }
            AudioChannelLayout::Surround71 => {
                unsafe {
                    // L_mono = T::from(0.5) * L + T::from(0.5) * R + 0.707 * C + T::from(0.5) * Ls + T::from(0.5) * Rs + T::from(0.5) * Lsr + T::from(0.5) * Rsr + T::from(0.5) * LFE
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..len {
                        *left.add(i) = T::from(0.5) * *left.add(i)
                            + T::from(0.5) * *right.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *lfe.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *rs.add(i)
                            + T::from(0.5) * *lsr.add(i)
                            + T::from(0.5) * *rsr.add(i);
                    }
                    self.layout = AudioChannelLayout::Mono;
                }
            }
        }
    }

    fn into_stereo(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {
                unsafe {
                    // L = R = Mono
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *left.add(i);
                    }
                    self.layout = AudioChannelLayout::Stereo;
                }
            }
            AudioChannelLayout::Stereo => {}
            AudioChannelLayout::Surround30 => {
                unsafe {
                    // L_stereo = L + 0.707 * C, R_stereo = R + 0.707 * C
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(INV_SQRT2) * *center.add(i);
                        *right.add(i) = *right.add(i) + T::from(INV_SQRT2) * *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Stereo;
                }
            }
            AudioChannelLayout::Surround40 => {
                unsafe {
                    // L_stereo = L + T::from(0.5) * Ls, R_stereo = R + T::from(0.5) * Rs
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(0.5) * *ls.add(i);
                        *right.add(i) = *right.add(i) + T::from(0.5) * *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Stereo;
                }
            }
            AudioChannelLayout::Surround51 => {
                unsafe {
                    // L_stereo = L + 0.707 * C + T::from(0.5) * Ls + T::from(0.5) * LFE
                    // R_stereo = R + 0.707 * C + T::from(0.5) * Rs + T::from(0.5) * LFE
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *lfe.add(i);
                        *right.add(i) = *right.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *rs.add(i)
                            + T::from(0.5) * *lfe.add(i);
                    }
                    self.layout = AudioChannelLayout::Stereo;
                }
            }
            AudioChannelLayout::Surround71 => {
                unsafe {
                    // L_stereo = L + 0.707 * C + T::from(0.5) * Ls + T::from(0.5) * Lsr + T::from(0.5) * LFE
                    // R_stereo = R + 0.707 * C + T::from(0.5) * Rs + T::from(0.5) * Rsr + T::from(0.5) * LFE
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *lsr.add(i)
                            + T::from(0.5) * *lfe.add(i);
                        *right.add(i) = *right.add(i)
                            + T::from(INV_SQRT2) * *center.add(i)
                            + T::from(0.5) * *rs.add(i)
                            + T::from(0.5) * *rsr.add(i)
                            + T::from(0.5) * *lfe.add(i);
                    }
                    self.layout = AudioChannelLayout::Stereo;
                }
            }
        }
    }

    /// Converts the buffer to Surround 3.0 (3 channels: L, R, C).
    fn into_surround30(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {
                unsafe {
                    // L = R = C = Mono
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *left.add(i);
                        *center.add(i) = *left.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround30;
                }
            }
            AudioChannelLayout::Stereo => {
                unsafe {
                    // L = L, R = R, C = T::from(0.5) * L + T::from(0.5) * R
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = T::from(0.5) * *left.add(i) + T::from(0.5) * *right.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround30;
                }
            }
            AudioChannelLayout::Surround30 => {}
            AudioChannelLayout::Surround40 => {
                unsafe {
                    // L = L + T::from(0.5) * Ls, R = R + T::from(0.5) * Rs, C = 0
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(0.5) * *ls.add(i);
                        *right.add(i) = *right.add(i) + T::from(0.5) * *rs.add(i);
                        *center.add(i) = T::from(0.0);
                    }
                    self.layout = AudioChannelLayout::Surround30;
                }
            }
            AudioChannelLayout::Surround51 => {
                unsafe {
                    // L = L + T::from(0.5) * Ls + T::from(0.5) * LFE, R = R + T::from(0.5) * Rs + T::from(0.5) * LFE, C = C
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) =
                            *left.add(i) + T::from(0.5) * *ls.add(i) + T::from(0.5) * *lfe.add(i);
                        *right.add(i) =
                            *right.add(i) + T::from(0.5) * *rs.add(i) + T::from(0.5) * *lfe.add(i);
                        *center.add(i) = *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround30;
                }
            }
            AudioChannelLayout::Surround71 => {
                unsafe {
                    // L = L + T::from(0.5) * Ls + T::from(0.5) * Lsr + T::from(0.5) * LFE
                    // R = R + T::from(0.5) * Rs + T::from(0.5) * Rsr + T::from(0.5) * LFE
                    // C = C
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i)
                            + T::from(0.5) * *ls.add(i)
                            + T::from(0.5) * *lsr.add(i)
                            + T::from(0.5) * *lfe.add(i);
                        *right.add(i) = *right.add(i)
                            + T::from(0.5) * *rs.add(i)
                            + T::from(0.5) * *rsr.add(i)
                            + T::from(0.5) * *lfe.add(i);
                        *center.add(i) = *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround30;
                }
            }
        }
    }

    /// Converts the buffer to Surround 4.0 (4 channels: L, R, Ls, Rs).
    fn into_surround40(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {
                unsafe {
                    // L = R = Ls = Rs = Mono
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    // let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    // let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *left.add(i);
                        *center.add(i) = *left.add(i);
                        *lfe.add(i) = *left.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround40;
                }
            }
            AudioChannelLayout::Stereo => {
                unsafe {
                    // L = L, R = R, Ls = L, Rs = R
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    // let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    // let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = *left.add(i);
                        *lfe.add(i) = *right.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround40;
                }
            }
            AudioChannelLayout::Surround30 => {
                unsafe {
                    // L = L, R = R, Ls = T::from(0.5) * C, Rs = T::from(0.5) * C
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    // let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    // let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = T::from(0.5) * *center.add(i);
                        *lfe.add(i) = T::from(0.5) * *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround40;
                }
            }
            AudioChannelLayout::Surround40 => {}
            AudioChannelLayout::Surround51 => {
                unsafe {
                    // L = L + T::from(0.5) * LFE, R = R + T::from(0.5) * LFE, Ls = Ls, Rs = Rs
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(0.5) * *lfe.add(i);
                        *right.add(i) = *right.add(i) + T::from(0.5) * *lfe.add(i);
                        *center.add(i) = *ls.add(i);
                        *lfe.add(i) = *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround40;
                }
            }
            AudioChannelLayout::Surround71 => {
                unsafe {
                    // L = L + T::from(0.5) * LFE, R = R + T::from(0.5) * LFE, Ls = Ls + T::from(0.5) * Lsr, Rs = Rs + T::from(0.5) * Rsr
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(0.5) * *lfe.add(i);
                        *right.add(i) = *right.add(i) + T::from(0.5) * *lfe.add(i);
                        *center.add(i) = *ls.add(i) + T::from(0.5) * *lsr.add(i);
                        *lfe.add(i) = *rs.add(i) + T::from(0.5) * *rsr.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround40;
                }
            }
        }
    }

    /// Converts the buffer to Surround 5.1 (6 channels: L, R, C, LFE, Ls, Rs).
    fn into_surround51(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {
                unsafe {
                    // L = R = C = Ls = Rs = Mono, LFE = 0
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *left.add(i);
                        *center.add(i) = *left.add(i);
                        *lfe.add(i) = T::from(0.0); // LFE
                        *ls.add(i) = *left.add(i);
                        *rs.add(i) = *left.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround51;
                }
            }
            AudioChannelLayout::Stereo => {
                unsafe {
                    // L = L, R = R, C = T::from(0.5) * L + T::from(0.5) * R, Ls = L, Rs = R, LFE = 0
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = T::from(0.5) * *left.add(i) + T::from(0.5) * *right.add(i);
                        *lfe.add(i) = T::from(0.0); // LFE
                        *ls.add(i) = *left.add(i);
                        *rs.add(i) = *right.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround51;
                }
            }
            AudioChannelLayout::Surround30 => {
                unsafe {
                    // L = L, R = R, C = C, Ls = T::from(0.5) * C, Rs = T::from(0.5) * C, LFE = 0
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = *center.add(i);
                        *lfe.add(i) = T::from(0.0); // LFE
                        *ls.add(i) = T::from(0.5) * *center.add(i);
                        *rs.add(i) = T::from(0.5) * *center.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround51;
                }
            }
            AudioChannelLayout::Surround40 => {
                unsafe {
                    // L = L, R = R, C = 0, LFE = 0, Ls = Ls, Rs = Rs
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    // let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    // let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i);
                        *right.add(i) = *right.add(i);
                        *center.add(i) = T::from(0.0); // C
                        *lfe.add(i) = T::from(0.0); // LFE
                        *ls.add(i) = *ls.add(i);
                        *rs.add(i) = *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround51;
                }
            }
            AudioChannelLayout::Surround51 => {}
            AudioChannelLayout::Surround71 => {
                unsafe {
                    // L = L + T::from(0.5) * Lsr, R = R + T::from(0.5) * Rsr, C = C, LFE = LFE, Ls = Ls, Rs = Rs
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();
                    for i in 0..self.len() {
                        *left.add(i) = *left.add(i) + T::from(0.5) * *lsr.add(i);
                        *right.add(i) = *right.add(i) + T::from(0.5) * *rsr.add(i);
                        *center.add(i) = *center.add(i);
                        *lfe.add(i) = *lfe.add(i);
                        *ls.add(i) = *ls.add(i);
                        *rs.add(i) = *rs.add(i);
                    }
                    self.layout = AudioChannelLayout::Surround51;
                }
            }
        }
    }

    /// Converts the buffer to Surround 7.1 (8 channels: L, R, C, LFE, Ls, Rs, Lsr, Rsr).
    fn into_surround71(&mut self) {
        match self.layout {
            AudioChannelLayout::Mono => {
                unsafe {
                    // L = R = C = Ls = Rs = Lsr = Rsr = Mono, LFE = 0
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::LeftSurround as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();

                    for i in 0..len {
                        let v = *left.add(i);
                        *left.add(i) = v;
                        *right.add(i) = v;
                        *center.add(i) = v;
                        *lfe.add(i) = T::from(0.0);
                        *ls.add(i) = v;
                        *rs.add(i) = v;
                        *lsr.add(i) = v;
                        *rsr.add(i) = v;
                    }
                }
                self.layout = AudioChannelLayout::Surround71;
            }

            AudioChannelLayout::Stereo => {
                unsafe {
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::LeftSurround as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();

                    for i in 0..len {
                        let l = *left.add(i);
                        let r = *right.add(i);

                        *center.add(i) = T::from(0.5) * l + T::from(0.5) * r;
                        *lfe.add(i) = T::from(0.0);

                        *ls.add(i) = l;
                        *rs.add(i) = r;
                        *lsr.add(i) = l;
                        *rsr.add(i) = r;
                    }
                }
                self.layout = AudioChannelLayout::Surround71;
            }

            AudioChannelLayout::Surround30 => {
                unsafe {
                    let len = self.len();
                    let aud = self.buffers.as_mut_ptr();

                    let left = (*aud.add(AudioChannel::Left as usize)).as_mut_ptr();
                    let right = (*aud.add(AudioChannel::Right as usize)).as_mut_ptr();
                    let center = (*aud.add(AudioChannel::Center as usize)).as_mut_ptr();
                    let lfe = (*aud.add(AudioChannel::LFE as usize)).as_mut_ptr();
                    let ls = (*aud.add(AudioChannel::LeftSurround as usize)).as_mut_ptr();
                    let rs = (*aud.add(AudioChannel::RightSurround as usize)).as_mut_ptr();
                    let lsr = (*aud.add(AudioChannel::RearLeft as usize)).as_mut_ptr();
                    let rsr = (*aud.add(AudioChannel::RearRight as usize)).as_mut_ptr();

                    for i in 0..len {
                        let c = *center.add(i);
                        *lfe.add(i) = T::from(0.0);
                        *ls.add(i) = T::from(0.5) * c;
                        *rs.add(i) = T::from(0.5) * c;
                        *lsr.add(i) = T::from(0.5) * c;
                        *rsr.add(i) = T::from(0.5) * c;
                    }
                }
                self.layout = AudioChannelLayout::Surround71;
            }

            AudioChannelLayout::Surround40 => {
                unsafe {
                    let aud = self.buffers.as_mut_ptr();

                    let center = &mut *aud.add(AudioChannel::Center as usize);
                    let lfe = &mut *aud.add(AudioChannel::LFE as usize);
                    let ls = &*aud.add(AudioChannel::LeftSurround as usize);
                    let rs = &*aud.add(AudioChannel::RightSurround as usize);
                    let lsr = &mut *aud.add(AudioChannel::RearLeft as usize);
                    let rsr = &mut *aud.add(AudioChannel::RearRight as usize);

                    center.fill(T::from(0.0));
                    lfe.fill(T::from(0.0));
                    lsr.copy_from_slice(ls);
                    rsr.copy_from_slice(rs);
                }
                self.layout = AudioChannelLayout::Surround71;
            }

            AudioChannelLayout::Surround51 => {
                unsafe {
                    let aud = self.buffers.as_mut_ptr();

                    let ls = &*aud.add(AudioChannel::LeftSurround as usize);
                    let rs = &*aud.add(AudioChannel::RightSurround as usize);
                    let lsr = &mut *aud.add(AudioChannel::RearLeft as usize);
                    let rsr = &mut *aud.add(AudioChannel::RearRight as usize);

                    lsr.copy_from_slice(ls);
                    rsr.copy_from_slice(rs);
                }
                self.layout = AudioChannelLayout::Surround71;
            }

            AudioChannelLayout::Surround71 => {}
        }
    }
}
