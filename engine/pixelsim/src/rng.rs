//! The only randomness in the crate. No `rand`, no clock.

/// xorshift64* stream. One per layer, so threading cannot perturb determinism.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1) // never zero
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// uniform in [0, 1)
    #[inline]
    pub fn f32(&mut self) -> f32 {
        // 24 mantissa bits taken from the high end of the word
        (self.next_u64() >> 40) as f32 * (1.0 / 16_777_216.0)
    }

    #[inline]
    pub fn bit(&mut self) -> bool {
        self.next_u64() >> 63 == 1
    }

    /// uniform in `0..n`, `0` when `n == 0`. Multiply-shift, no division.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    /// `true` with probability `p255 / 256`
    #[inline]
    pub fn chance_u8(&mut self, p255: u8) -> bool {
        p255 != 0 && ((self.next_u64() >> 56) as u8) < p255
    }
}

/// Stateless, order-independent per-cell randomness (splitmix64 finaliser over the tuple).
#[inline]
pub fn hash_rng(seed: u64, tick: u64, x: u16, y: u16) -> u64 {
    let mut z = seed
        ^ tick.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (x as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ (y as u64).wrapping_mul(0xA076_1D64_78BD_642F);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
