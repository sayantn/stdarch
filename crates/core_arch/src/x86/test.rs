//! Utilities used in testing the x86 intrinsics

use crate::core_arch::x86::*;
use std::{intrinsics::simd::*, mem::transmute};

#[track_caller]
#[target_feature(enable = "sse2")]
pub fn assert_eq_m128i(a: __m128i, b: __m128i) {
    assert_eq!(a.as_array(), b.as_array())
}

#[track_caller]
#[target_feature(enable = "sse2")]
pub fn assert_eq_m128d(a: __m128d, b: __m128d) {
    if _mm_movemask_pd(_mm_cmpeq_pd(a, b)) != 0b11 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "sse2")]
pub fn get_m128d(a: __m128d, idx: usize) -> f64 {
    a.as_array()[idx]
}

#[track_caller]
#[target_feature(enable = "sse")]
pub fn assert_eq_m128(a: __m128, b: __m128) {
    let r = _mm_cmpeq_ps(a, b);
    if _mm_movemask_ps(r) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "sse")]
pub fn get_m128(a: __m128, idx: usize) -> f32 {
    a.as_array()[idx]
}

#[track_caller]
#[target_feature(enable = "avx512fp16,avx512vl")]
pub fn assert_eq_m128h(a: __m128h, b: __m128h) {
    let r = _mm_cmp_ph_mask::<_CMP_EQ_OQ>(a, b);
    if r != 0b1111_1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

// not actually an intrinsic but useful in various tests as we proted from
// `i64x2::new` which is backwards from `_mm_set_epi64x`
#[target_feature(enable = "sse2")]
pub fn _mm_setr_epi64x(a: i64, b: i64) -> __m128i {
    _mm_set_epi64x(b, a)
}

#[track_caller]
#[target_feature(enable = "avx")]
pub fn assert_eq_m256i(a: __m256i, b: __m256i) {
    assert_eq!(a.as_array(), b.as_array())
}

#[track_caller]
#[target_feature(enable = "avx")]
pub fn assert_eq_m256d(a: __m256d, b: __m256d) {
    let cmp = _mm256_cmp_pd::<_CMP_EQ_OQ>(a, b);
    if _mm256_movemask_pd(cmp) != 0b1111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "avx")]
pub fn get_m256d(a: __m256d, idx: usize) -> f64 {
    a.as_array()[idx]
}

#[track_caller]
#[target_feature(enable = "avx")]
pub fn assert_eq_m256(a: __m256, b: __m256) {
    let cmp = _mm256_cmp_ps::<_CMP_EQ_OQ>(a, b);
    if _mm256_movemask_ps(cmp) != 0b11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "avx")]
pub fn get_m256(a: __m256, idx: usize) -> f32 {
    a.as_array()[idx]
}

#[track_caller]
#[target_feature(enable = "avx512fp16,avx512vl")]
pub fn assert_eq_m256h(a: __m256h, b: __m256h) {
    let r = _mm256_cmp_ph_mask::<_CMP_EQ_OQ>(a, b);
    if r != 0b11111111_11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[target_feature(enable = "avx512f")]
pub fn get_m512(a: __m512, idx: usize) -> f32 {
    a.as_array()[idx]
}

#[target_feature(enable = "avx512f")]
pub fn get_m512d(a: __m512d, idx: usize) -> f64 {
    a.as_array()[idx]
}

#[target_feature(enable = "avx512f")]
pub fn get_m512i(a: __m512i, idx: usize) -> i64 {
    a.as_array()[idx]
}

// These intrinsics doesn't exist on x86 b/c it requires a 64-bit register,
// which doesn't exist on x86!
#[cfg(target_arch = "x86")]
mod x86_polyfill {
    use crate::core_arch::x86::*;
    use crate::intrinsics::simd::*;

    #[target_feature(enable = "sse4.1")]
    #[rustc_legacy_const_generics(2)]
    pub fn _mm_insert_epi64<const INDEX: i32>(a: __m128i, val: i64) -> __m128i {
        static_assert_uimm_bits!(INDEX, 1);
        unsafe { transmute(simd_insert!(a.as_i64x2(), INDEX as u32, val)) }
    }

    #[target_feature(enable = "avx")]
    #[rustc_legacy_const_generics(2)]
    pub fn _mm256_insert_epi64<const INDEX: i32>(a: __m256i, val: i64) -> __m256i {
        static_assert_uimm_bits!(INDEX, 2);
        unsafe { transmute(simd_insert!(a.as_i64x4(), INDEX as u32, val)) }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_polyfill {
    pub use crate::core_arch::x86_64::{_mm_insert_epi64, _mm256_insert_epi64};
}
pub use self::x86_polyfill::*;

#[track_caller]
#[target_feature(enable = "avx512f")]
pub fn assert_eq_m512i(a: __m512i, b: __m512i) {
    assert_eq!(a.as_array(), b.as_array())
}

#[track_caller]
#[target_feature(enable = "avx512f")]
pub fn assert_eq_m512(a: __m512, b: __m512) {
    let cmp = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(a, b);
    if cmp != 0b11111111_11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "avx512f")]
pub fn assert_eq_m512d(a: __m512d, b: __m512d) {
    let cmp = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(a, b);
    if cmp != 0b11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}

#[track_caller]
#[target_feature(enable = "avx512fp16")]
pub fn assert_eq_m512h(a: __m512h, b: __m512h) {
    let r = _mm512_cmp_ph_mask::<_CMP_EQ_OQ>(a, b);
    if r != 0b11111111_11111111_11111111_11111111 {
        panic!("{:?} != {:?}", a, b);
    }
}
