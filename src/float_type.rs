use std::ops::{Add, AddAssign, Div, Mul, Sub};

/// needs to be optimized in a future
pub trait FloatType:
    PartialOrd
    + Clone
    + Copy
    + Add<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Sub<Output = Self>
    + From<f32>
    + AddAssign
    + std::iter::Sum
{
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn to_usize(self) -> usize;
    fn to_isize(self) -> isize;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_nan(self) -> bool;
    fn from_usize(value: usize) -> Self;
    fn from_isize(value: isize) -> Self;
    fn max(self, other: Self) -> Self;
    fn log10(self) -> Self;

    fn from_f64(n: f64) -> Self;
    fn from_f32(n: f32) -> Self;

    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;

    fn ceil(self) -> Self;
    fn floor(self) -> Self;
}

impl FloatType for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        return self;
    }

    #[inline]
    fn to_f64(self) -> f64 {
        return self as f64;
    }

    #[inline]
    fn is_nan(self) -> bool {
        return self.is_nan();
    }

    #[inline]
    fn abs(self) -> f32 {
        return self.abs();
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        return value as Self;
    }

    #[inline]
    fn sqrt(self) -> Self {
        return self.sqrt();
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        return self.max(other);
    }

    #[inline]
    fn log10(self) -> Self {
        return self.log10();
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        return self.powf(n);
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        return self.powi(n);
    }

    #[inline]
    fn from_f64(n: f64) -> Self {
        return n as Self;
    }

    #[inline]
    fn ceil(self) -> Self {
        return self.ceil();
    }

    #[inline]
    fn to_usize(self) -> usize {
        return self as usize;
    }

    #[inline]
    fn from_f32(n: f32) -> Self {
        return n as Self;
    }

    #[inline]
    fn floor(self) -> Self {
        return self.floor();
    }

    #[inline]
    fn to_isize(self) -> isize {
        return self as isize;
    }

    #[inline]
    fn from_isize(value: isize) -> Self {
        return value as Self;
    }
}

impl FloatType for f64 {
    #[inline]
    fn to_f32(self) -> f32 {
        return self as f32;
    }

    #[inline]
    fn to_f64(self) -> f64 {
        return self;
    }

    #[inline]
    fn is_nan(self) -> bool {
        return self.is_nan();
    }

    #[inline]
    fn abs(self) -> f64 {
        return self.abs();
    }

    #[inline]
    fn from_usize(value: usize) -> Self {
        return value as Self;
    }

    #[inline]
    fn sqrt(self) -> Self {
        return self.sqrt();
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        return self.max(other);
    }

    #[inline]
    fn log10(self) -> Self {
        return self.log10();
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        return self.powf(n);
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        return self.powi(n);
    }

    #[inline]
    fn from_f64(n: f64) -> Self {
        return n as Self;
    }

    #[inline]
    fn ceil(self) -> Self {
        return self.ceil();
    }

    #[inline]
    fn to_usize(self) -> usize {
        return self as usize;
    }

    #[inline]
    fn from_f32(n: f32) -> Self {
        return n as Self;
    }

    #[inline]
    fn floor(self) -> Self {
        return self.floor();
    }

    #[inline]
    fn to_isize(self) -> isize {
        return self as isize;
    }

    #[inline]
    fn from_isize(value: isize) -> Self {
        return value as Self;
    }
}
