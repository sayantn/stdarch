// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen2/spec/` and run the following command to re-generate this file:
//
// ```
// cargo run --bin=stdarch-gen2 -- crates/stdarch-gen2/spec
// ```
#![allow(improper_ctypes)]

#[cfg(test)]
use stdarch_test::assert_instr;

use super::*;

#[doc = "Signed Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sabal))]
pub unsafe fn vabal_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let d: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: int8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: int8x8_t = vabd_s8(d, e);
    let f: uint8x8_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

#[doc = "Signed Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sabal))]
pub unsafe fn vabal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let d: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let e: int16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    let f: int16x4_t = vabd_s16(d, e);
    let f: uint16x4_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

#[doc = "Signed Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sabal))]
pub unsafe fn vabal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let d: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    let e: int32x2_t = simd_shuffle!(c, c, [2, 3]);
    let f: int32x2_t = vabd_s32(d, e);
    let f: uint32x2_t = simd_cast(f);
    simd_add(a, simd_cast(f))
}

#[doc = "Unsigned Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uabal))]
pub unsafe fn vabal_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let d: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: uint8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: uint8x8_t = vabd_u8(d, e);
    simd_add(a, simd_cast(f))
}

#[doc = "Unsigned Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uabal))]
pub unsafe fn vabal_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let d: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let e: uint16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    let f: uint16x4_t = vabd_u16(d, e);
    simd_add(a, simd_cast(f))
}

#[doc = "Unsigned Absolute difference and Accumulate Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabal_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uabal))]
pub unsafe fn vabal_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let d: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    let e: uint32x2_t = simd_shuffle!(c, c, [2, 3]);
    let f: uint32x2_t = vabd_u32(d, e);
    simd_add(a, simd_cast(f))
}

#[doc = "Absolute difference between the arguments of Floating"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabd_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fabd.v1f64"
        )]
        fn _vabd_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vabd_f64(a, b)
}

#[doc = "Absolute difference between the arguments of Floating"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabdq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fabd.v2f64"
        )]
        fn _vabdq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vabdq_f64(a, b)
}

#[doc = "Floating-point absolute difference"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabdd_f64(a: f64, b: f64) -> f64 {
    simd_extract!(vabd_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point absolute difference"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabds_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fabd))]
pub unsafe fn vabds_f32(a: f32, b: f32) -> f32 {
    simd_extract!(vabd_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Signed Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let d: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let e: uint16x4_t = simd_cast(vabd_s16(c, d));
    simd_cast(e)
}

#[doc = "Signed Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let d: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    let e: uint32x2_t = simd_cast(vabd_s32(c, d));
    simd_cast(e)
}

#[doc = "Signed Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sabdl))]
pub unsafe fn vabdl_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let e: uint8x8_t = simd_cast(vabd_s8(c, d));
    simd_cast(e)
}

#[doc = "Unsigned Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vabdl_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_cast(vabd_u8(c, d))
}

#[doc = "Unsigned Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vabdl_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let d: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    simd_cast(vabd_u16(c, d))
}

#[doc = "Unsigned Absolute difference Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabdl_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uabdl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vabdl_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle!(a, a, [2, 3]);
    let d: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    simd_cast(vabd_u32(c, d))
}

#[doc = "Floating-point absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabs_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vabs_f64(a: float64x1_t) -> float64x1_t {
    simd_fabs(a)
}

#[doc = "Floating-point absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vabsq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fabs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vabsq_f64(a: float64x2_t) -> float64x2_t {
    simd_fabs(a)
}

#[doc = "Add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vaddd_s64(a: i64, b: i64) -> i64 {
    a.wrapping_add(b)
}

#[doc = "Add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vaddd_u64(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

#[doc = "Signed Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlv_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlv_s16(a: int16x4_t) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.saddlv.i32.v4i16"
        )]
        fn _vaddlv_s16(a: int16x4_t) -> i32;
    }
    _vaddlv_s16(a)
}

#[doc = "Signed Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlvq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlvq_s16(a: int16x8_t) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.saddlv.i32.v8i16"
        )]
        fn _vaddlvq_s16(a: int16x8_t) -> i32;
    }
    _vaddlvq_s16(a)
}

#[doc = "Signed Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlvq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(saddlv))]
pub unsafe fn vaddlvq_s32(a: int32x4_t) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.saddlv.i64.v4i32"
        )]
        fn _vaddlvq_s32(a: int32x4_t) -> i64;
    }
    _vaddlvq_s32(a)
}

#[doc = "Signed Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlv_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(saddlp))]
pub unsafe fn vaddlv_s32(a: int32x2_t) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.saddlv.i64.v2i32"
        )]
        fn _vaddlv_s32(a: int32x2_t) -> i64;
    }
    _vaddlv_s32(a)
}

#[doc = "Unsigned Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlv_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlv_u16(a: uint16x4_t) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uaddlv.i32.v4i16"
        )]
        fn _vaddlv_u16(a: int16x4_t) -> i32;
    }
    _vaddlv_u16(a.as_signed()).as_unsigned()
}

#[doc = "Unsigned Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlvq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlvq_u16(a: uint16x8_t) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uaddlv.i32.v8i16"
        )]
        fn _vaddlvq_u16(a: int16x8_t) -> i32;
    }
    _vaddlvq_u16(a.as_signed()).as_unsigned()
}

#[doc = "Unsigned Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlvq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uaddlv))]
pub unsafe fn vaddlvq_u32(a: uint32x4_t) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uaddlv.i64.v4i32"
        )]
        fn _vaddlvq_u32(a: int32x4_t) -> i64;
    }
    _vaddlvq_u32(a.as_signed()).as_unsigned()
}

#[doc = "Unsigned Add Long across Vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddlv_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uaddlp))]
pub unsafe fn vaddlv_u32(a: uint32x2_t) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uaddlv.i64.v2i32"
        )]
        fn _vaddlv_u32(a: int32x2_t) -> i64;
    }
    _vaddlv_u32(a.as_signed()).as_unsigned()
}

#[doc = "Floating-point add across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddv_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(faddp))]
pub unsafe fn vaddv_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.faddv.f32.v2f32"
        )]
        fn _vaddv_f32(a: float32x2_t) -> f32;
    }
    _vaddv_f32(a)
}

#[doc = "Floating-point add across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddvq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(faddp))]
pub unsafe fn vaddvq_f32(a: float32x4_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.faddv.f32.v4f32"
        )]
        fn _vaddvq_f32(a: float32x4_t) -> f32;
    }
    _vaddvq_f32(a)
}

#[doc = "Floating-point add across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vaddvq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(faddp))]
pub unsafe fn vaddvq_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.faddv.f64.v2f64"
        )]
        fn _vaddvq_f64(a: float64x2_t) -> f64;
    }
    _vaddvq_f64(a)
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxs.v16i8"
        )]
        fn _vbcaxq_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
    }
    _vbcaxq_s8(a, b, c)
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxs.v8i16"
        )]
        fn _vbcaxq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _vbcaxq_s16(a, b, c)
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxs.v4i32"
        )]
        fn _vbcaxq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vbcaxq_s32(a, b, c)
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxs.v2i64"
        )]
        fn _vbcaxq_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _vbcaxq_s64(a, b, c)
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxu.v16i8"
        )]
        fn _vbcaxq_u8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
    }
    _vbcaxq_u8(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_u16(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxu.v8i16"
        )]
        fn _vbcaxq_u16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _vbcaxq_u16(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxu.v4i32"
        )]
        fn _vbcaxq_u32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vbcaxq_u32(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Bit clear and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vbcaxq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(bcax))]
pub unsafe fn vbcaxq_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.bcaxu.v2i64"
        )]
        fn _vbcaxq_u64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _vbcaxq_u64(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcadd_rot270_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcadd_rot270_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot270.v2f32"
        )]
        fn _vcadd_rot270_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    _vcadd_rot270_f32(a, b)
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaddq_rot270_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcaddq_rot270_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot270.v4f32"
        )]
        fn _vcaddq_rot270_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vcaddq_rot270_f32(a, b)
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaddq_rot270_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcaddq_rot270_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot270.v2f64"
        )]
        fn _vcaddq_rot270_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vcaddq_rot270_f64(a, b)
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcadd_rot90_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcadd_rot90_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot90.v2f32"
        )]
        fn _vcadd_rot90_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    _vcadd_rot90_f32(a, b)
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaddq_rot90_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcaddq_rot90_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot90.v4f32"
        )]
        fn _vcaddq_rot90_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vcaddq_rot90_f32(a, b)
}

#[doc = "Floating-point complex add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaddq_rot90_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcadd))]
pub unsafe fn vcaddq_rot90_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcadd.rot90.v2f64"
        )]
        fn _vcaddq_rot90_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vcaddq_rot90_f64(a, b)
}

#[doc = "Floating-point absolute compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcage_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcage_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facge.v1i64.v1f64"
        )]
        fn _vcage_f64(a: float64x1_t, b: float64x1_t) -> int64x1_t;
    }
    _vcage_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcageq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcageq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facge.v2i64.v2f64"
        )]
        fn _vcageq_f64(a: float64x2_t, b: float64x2_t) -> int64x2_t;
    }
    _vcageq_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaged_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcaged_f64(a: f64, b: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facge.i64.f64"
        )]
        fn _vcaged_f64(a: f64, b: f64) -> i64;
    }
    _vcaged_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcages_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcages_f32(a: f32, b: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facge.i32.f32"
        )]
        fn _vcages_f32(a: f32, b: f32) -> i32;
    }
    _vcages_f32(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcagt_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcagt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facgt.v1i64.v1f64"
        )]
        fn _vcagt_f64(a: float64x1_t, b: float64x1_t) -> int64x1_t;
    }
    _vcagt_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcagtq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcagtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facgt.v2i64.v2f64"
        )]
        fn _vcagtq_f64(a: float64x2_t, b: float64x2_t) -> int64x2_t;
    }
    _vcagtq_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcagtd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcagtd_f64(a: f64, b: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facgt.i64.f64"
        )]
        fn _vcagtd_f64(a: f64, b: f64) -> i64;
    }
    _vcagtd_f64(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcagts_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcagts_f32(a: f32, b: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.facgt.i32.f32"
        )]
        fn _vcagts_f32(a: f32, b: f32) -> i32;
    }
    _vcagts_f32(a, b).as_unsigned()
}

#[doc = "Floating-point absolute compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcale_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcale_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    vcage_f64(b, a)
}

#[doc = "Floating-point absolute compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaleq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcaleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    vcageq_f64(b, a)
}

#[doc = "Floating-point absolute compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaled_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcaled_f64(a: f64, b: f64) -> u64 {
    vcaged_f64(b, a)
}

#[doc = "Floating-point absolute compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcales_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcales_f32(a: f32, b: f32) -> u32 {
    vcages_f32(b, a)
}

#[doc = "Floating-point absolute compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcalt_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcalt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    vcagt_f64(b, a)
}

#[doc = "Floating-point absolute compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaltq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcaltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    vcagtq_f64(b, a)
}

#[doc = "Floating-point absolute compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcaltd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcaltd_f64(a: f64, b: f64) -> u64 {
    vcagtd_f64(b, a)
}

#[doc = "Floating-point absolute compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcalts_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(facgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcalts_f32(a: f32, b: f32) -> u32 {
    vcagts_f32(b, a)
}

#[doc = "Floating-point compare equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceq_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

#[doc = "Floating-point compare equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceq_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceq_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceq_p64(a: poly64x1_t, b: poly64x1_t) -> uint64x1_t {
    simd_eq(a, b)
}

#[doc = "Compare bitwise Equal (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqq_p64(a: poly64x2_t, b: poly64x2_t) -> uint64x2_t {
    simd_eq(a, b)
}

#[doc = "Floating-point compare equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqd_f64(a: f64, b: f64) -> u64 {
    simd_extract!(vceq_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point compare equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqs_f32(a: f32, b: f32) -> u32 {
    simd_extract!(vceq_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Compare bitwise equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqd_s64(a: i64, b: i64) -> u64 {
    transmute(vceq_s64(transmute(a), transmute(b)))
}

#[doc = "Compare bitwise equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqd_u64(a: u64, b: u64) -> u64 {
    transmute(vceq_u64(transmute(a), transmute(b)))
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_eq(a, transmute(b))
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_eq(a, transmute(b))
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_eq(a, transmute(b))
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_p8(a: poly8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_p8(a: poly8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_p64(a: poly64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_eq(a, transmute(b))
}

#[doc = "Signed compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_p64(a: poly64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_u8(a: uint8x8_t) -> uint8x8_t {
    let b: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_u8(a: uint8x16_t) -> uint8x16_t {
    let b: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_u16(a: uint16x4_t) -> uint16x4_t {
    let b: u16x4 = u16x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_u16(a: uint16x8_t) -> uint16x8_t {
    let b: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_u32(a: uint32x2_t) -> uint32x2_t {
    let b: u32x2 = u32x2::new(0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_u32(a: uint32x4_t) -> uint32x4_t {
    let b: u32x4 = u32x4::new(0, 0, 0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqz_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqz_u64(a: uint64x1_t) -> uint64x1_t {
    let b: u64x1 = u64x1::new(0);
    simd_eq(a, transmute(b))
}

#[doc = "Unsigned compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmeq))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzq_u64(a: uint64x2_t) -> uint64x2_t {
    let b: u64x2 = u64x2::new(0, 0);
    simd_eq(a, transmute(b))
}

#[doc = "Compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzd_s64(a: i64) -> u64 {
    transmute(vceqz_s64(transmute(a)))
}

#[doc = "Compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzd_u64(a: u64) -> u64 {
    transmute(vceqz_u64(transmute(a)))
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzs_f32(a: f32) -> u32 {
    simd_extract!(vceqz_f32(vdup_n_f32(a)), 0)
}

#[doc = "Floating-point compare bitwise equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vceqzd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vceqzd_f64(a: f64) -> u64 {
    simd_extract!(vceqz_f64(vdup_n_f64(a)), 0)
}

#[doc = "Floating-point compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcge_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcge_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

#[doc = "Floating-point compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgeq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgeq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

#[doc = "Compare signed greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcge_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcge_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

#[doc = "Compare signed greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgeq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgeq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

#[doc = "Compare unsigned greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcge_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcge_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_ge(a, b)
}

#[doc = "Compare unsigned greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgeq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgeq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_ge(a, b)
}

#[doc = "Floating-point compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcged_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcged_f64(a: f64, b: f64) -> u64 {
    simd_extract!(vcge_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcges_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcges_f32(a: f32, b: f32) -> u32 {
    simd_extract!(vcge_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcged_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcged_s64(a: i64, b: i64) -> u64 {
    transmute(vcge_s64(transmute(a), transmute(b)))
}

#[doc = "Compare greater than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcged_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcged_u64(a: u64, b: u64) -> u64 {
    transmute(vcge_u64(transmute(a), transmute(b)))
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_ge(a, transmute(b))
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_ge(a, transmute(b))
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_ge(a, transmute(b))
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgez_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgez_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_ge(a, transmute(b))
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_ge(a, transmute(b))
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezd_f64(a: f64) -> u64 {
    simd_extract!(vcgez_f64(vdup_n_f64(a)), 0)
}

#[doc = "Floating-point compare greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezs_f32(a: f32) -> u32 {
    simd_extract!(vcgez_f32(vdup_n_f32(a)), 0)
}

#[doc = "Compare signed greater than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgezd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgezd_s64(a: i64) -> u64 {
    transmute(vcgez_s64(transmute(a)))
}

#[doc = "Floating-point compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgt_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

#[doc = "Floating-point compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

#[doc = "Compare signed greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgt_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

#[doc = "Compare signed greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

#[doc = "Compare unsigned greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgt_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_gt(a, b)
}

#[doc = "Compare unsigned greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_gt(a, b)
}

#[doc = "Floating-point compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtd_f64(a: f64, b: f64) -> u64 {
    simd_extract!(vcgt_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgts_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgts_f32(a: f32, b: f32) -> u32 {
    simd_extract!(vcgt_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtd_s64(a: i64, b: i64) -> u64 {
    transmute(vcgt_s64(transmute(a), transmute(b)))
}

#[doc = "Compare greater than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtd_u64(a: u64, b: u64) -> u64 {
    transmute(vcgt_u64(transmute(a), transmute(b)))
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_gt(a, transmute(b))
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_gt(a, transmute(b))
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_gt(a, transmute(b))
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtz_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_gt(a, transmute(b))
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_gt(a, transmute(b))
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzd_f64(a: f64) -> u64 {
    simd_extract!(vcgtz_f64(vdup_n_f64(a)), 0)
}

#[doc = "Floating-point compare greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzs_f32(a: f32) -> u32 {
    simd_extract!(vcgtz_f32(vdup_n_f32(a)), 0)
}

#[doc = "Compare signed greater than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcgtzd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcgtzd_s64(a: i64) -> u64 {
    transmute(vcgtz_s64(transmute(a)))
}

#[doc = "Floating-point compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcle_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcle_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

#[doc = "Floating-point compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcleq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcleq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

#[doc = "Compare signed less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcle_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcle_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

#[doc = "Compare signed less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcleq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmge))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcleq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

#[doc = "Compare unsigned less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcle_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcle_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_le(a, b)
}

#[doc = "Compare unsigned less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcleq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcleq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_le(a, b)
}

#[doc = "Floating-point compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcled_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcled_f64(a: f64, b: f64) -> u64 {
    simd_extract!(vcle_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcles_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcles_f32(a: f32, b: f32) -> u32 {
    simd_extract!(vcle_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcled_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcled_u64(a: u64, b: u64) -> u64 {
    transmute(vcle_u64(transmute(a), transmute(b)))
}

#[doc = "Compare less than or equal"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcled_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcled_s64(a: i64, b: i64) -> u64 {
    transmute(vcle_s64(transmute(a), transmute(b)))
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_le(a, transmute(b))
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_le(a, transmute(b))
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_le(a, transmute(b))
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclez_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclez_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_le(a, transmute(b))
}

#[doc = "Compare signed less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmle))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_le(a, transmute(b))
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezd_f64(a: f64) -> u64 {
    simd_extract!(vclez_f64(vdup_n_f64(a)), 0)
}

#[doc = "Floating-point compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezs_f32(a: f32) -> u32 {
    simd_extract!(vclez_f32(vdup_n_f32(a)), 0)
}

#[doc = "Compare less than or equal to zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclezd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclezd_s64(a: i64) -> u64 {
    transmute(vclez_s64(transmute(a)))
}

#[doc = "Floating-point compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclt_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclt_f64(a: float64x1_t, b: float64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

#[doc = "Floating-point compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltq_f64(a: float64x2_t, b: float64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

#[doc = "Compare signed less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclt_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclt_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

#[doc = "Compare signed less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmgt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

#[doc = "Compare unsigned less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclt_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclt_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_lt(a, b)
}

#[doc = "Compare unsigned less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmhi))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_lt(a, b)
}

#[doc = "Compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltd_u64(a: u64, b: u64) -> u64 {
    transmute(vclt_u64(transmute(a), transmute(b)))
}

#[doc = "Compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltd_s64(a: i64, b: i64) -> u64 {
    transmute(vclt_s64(transmute(a), transmute(b)))
}

#[doc = "Floating-point compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vclts_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vclts_f32(a: f32, b: f32) -> u32 {
    simd_extract!(vclt_f32(vdup_n_f32(a), vdup_n_f32(b)), 0)
}

#[doc = "Floating-point compare less than"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltd_f64(a: f64, b: f64) -> u64 {
    simd_extract!(vclt_f64(vdup_n_f64(a), vdup_n_f64(b)), 0)
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_f32(a: float32x2_t) -> uint32x2_t {
    let b: f32x2 = f32x2::new(0.0, 0.0);
    simd_lt(a, transmute(b))
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_f32(a: float32x4_t) -> uint32x4_t {
    let b: f32x4 = f32x4::new(0.0, 0.0, 0.0, 0.0);
    simd_lt(a, transmute(b))
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_f64(a: float64x1_t) -> uint64x1_t {
    let b: f64 = 0.0;
    simd_lt(a, transmute(b))
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_f64(a: float64x2_t) -> uint64x2_t {
    let b: f64x2 = f64x2::new(0.0, 0.0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_s8(a: int8x8_t) -> uint8x8_t {
    let b: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_s8(a: int8x16_t) -> uint8x16_t {
    let b: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_s16(a: int16x4_t) -> uint16x4_t {
    let b: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_s16(a: int16x8_t) -> uint16x8_t {
    let b: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_s32(a: int32x2_t) -> uint32x2_t {
    let b: i32x2 = i32x2::new(0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_s32(a: int32x4_t) -> uint32x4_t {
    let b: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltz_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltz_s64(a: int64x1_t) -> uint64x1_t {
    let b: i64x1 = i64x1::new(0);
    simd_lt(a, transmute(b))
}

#[doc = "Compare signed less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmlt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzq_s64(a: int64x2_t) -> uint64x2_t {
    let b: i64x2 = i64x2::new(0, 0);
    simd_lt(a, transmute(b))
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzd_f64(a: f64) -> u64 {
    simd_extract!(vcltz_f64(vdup_n_f64(a)), 0)
}

#[doc = "Floating-point compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzs_f32(a: f32) -> u32 {
    simd_extract!(vcltz_f32(vdup_n_f32(a)), 0)
}

#[doc = "Compare less than zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcltzd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(asr))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcltzd_s64(a: i64) -> u64 {
    transmute(vcltz_s64(transmute(a)))
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmla_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot0.v2f32"
        )]
        fn _vcmla_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t;
    }
    _vcmla_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot0.v4f32"
        )]
        fn _vcmlaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
    }
    _vcmlaq_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot0.v2f64"
        )]
        fn _vcmlaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    _vcmlaq_f64(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert!(LANE == 0);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert!(LANE == 0);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot180_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmla_rot180_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot180.v2f32"
        )]
        fn _vcmla_rot180_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t;
    }
    _vcmla_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot180_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot180_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot180.v4f32"
        )]
        fn _vcmlaq_rot180_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
    }
    _vcmlaq_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot180_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot180_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot180.v2f64"
        )]
        fn _vcmlaq_rot180_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    _vcmlaq_rot180_f64(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot180_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot180_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert!(LANE == 0);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot180_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot180_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert!(LANE == 0);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot180_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot180_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot180_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot180_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot180_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot270_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmla_rot270_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot270.v2f32"
        )]
        fn _vcmla_rot270_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t;
    }
    _vcmla_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot270_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot270_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot270.v4f32"
        )]
        fn _vcmlaq_rot270_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
    }
    _vcmlaq_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot270_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot270_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot270.v2f64"
        )]
        fn _vcmlaq_rot270_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    _vcmlaq_rot270_f64(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot270_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot270_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert!(LANE == 0);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot270_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot270_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert!(LANE == 0);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot270_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot270_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot270_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot270_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot270_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot90_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmla_rot90_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot90.v2f32"
        )]
        fn _vcmla_rot90_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t;
    }
    _vcmla_rot90_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot90_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot90_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot90.v4f32"
        )]
        fn _vcmlaq_rot90_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
    }
    _vcmlaq_rot90_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot90_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
#[cfg_attr(test, assert_instr(fcmla))]
pub unsafe fn vcmlaq_rot90_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcmla.rot90.v2f64"
        )]
        fn _vcmlaq_rot90_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    _vcmlaq_rot90_f64(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot90_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot90_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert!(LANE == 0);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot90_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot90_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot90_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert!(LANE == 0);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot90_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmla_rot90_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmla_rot90_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x2_t = simd_shuffle!(c, c, [2 * LANE as u32, 2 * LANE as u32 + 1]);
    vcmla_rot90_f32(a, b, c)
}

#[doc = "Floating-point complex multiply accumulate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcmlaq_rot90_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,fcma")]
#[cfg_attr(test, assert_instr(fcmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_fcma", issue = "117222")]
pub unsafe fn vcmlaq_rot90_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: float32x4_t = simd_shuffle!(
        c,
        c,
        [
            2 * LANE as u32,
            2 * LANE as u32 + 1,
            2 * LANE as u32,
            2 * LANE as u32 + 1
        ]
    );
    vcmlaq_rot90_f32(a, b, c)
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_f32<const LANE1: i32, const LANE2: i32>(
    a: float32x2_t,
    b: float32x2_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_s8<const LANE1: i32, const LANE2: i32>(
    a: int8x8_t,
    b: int8x8_t,
) -> int8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_s8<const LANE1: i32, const LANE2: i32>(
    a: int8x16_t,
    b: int8x8_t,
) -> int8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 3);
    let b: int8x16_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_s16<const LANE1: i32, const LANE2: i32>(
    a: int16x4_t,
    b: int16x4_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_s16<const LANE1: i32, const LANE2: i32>(
    a: int16x8_t,
    b: int16x4_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 2);
    let b: int16x8_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_s32<const LANE1: i32, const LANE2: i32>(
    a: int32x2_t,
    b: int32x2_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_s32<const LANE1: i32, const LANE2: i32>(
    a: int32x4_t,
    b: int32x2_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 1);
    let b: int32x4_t = simd_shuffle!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_u8<const LANE1: i32, const LANE2: i32>(
    a: uint8x8_t,
    b: uint8x8_t,
) -> uint8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_u8<const LANE1: i32, const LANE2: i32>(
    a: uint8x16_t,
    b: uint8x8_t,
) -> uint8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 3);
    let b: uint8x16_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_u16<const LANE1: i32, const LANE2: i32>(
    a: uint16x4_t,
    b: uint16x4_t,
) -> uint16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_u16<const LANE1: i32, const LANE2: i32>(
    a: uint16x8_t,
    b: uint16x4_t,
) -> uint16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 2);
    let b: uint16x8_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_u32<const LANE1: i32, const LANE2: i32>(
    a: uint32x2_t,
    b: uint32x2_t,
) -> uint32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_u32<const LANE1: i32, const LANE2: i32>(
    a: uint32x4_t,
    b: uint32x2_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 1);
    let b: uint32x4_t = simd_shuffle!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_p8<const LANE1: i32, const LANE2: i32>(
    a: poly8x8_t,
    b: poly8x8_t,
) -> poly8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_p8<const LANE1: i32, const LANE2: i32>(
    a: poly8x16_t,
    b: poly8x8_t,
) -> poly8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 3);
    let b: poly8x16_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_lane_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_lane_p16<const LANE1: i32, const LANE2: i32>(
    a: poly16x4_t,
    b: poly16x4_t,
) -> poly16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_p16<const LANE1: i32, const LANE2: i32>(
    a: poly16x8_t,
    b: poly16x4_t,
) -> poly16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 2);
    let b: poly16x8_t = simd_shuffle!(b, b, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_f32<const LANE1: i32, const LANE2: i32>(
    a: float32x2_t,
    b: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 2);
    let a: float32x4_t = simd_shuffle!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_f32<const LANE1: i32, const LANE2: i32>(
    a: float32x4_t,
    b: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_f64<const LANE1: i32, const LANE2: i32>(
    a: float64x2_t,
    b: float64x2_t,
) -> float64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_s8<const LANE1: i32, const LANE2: i32>(
    a: int8x8_t,
    b: int8x16_t,
) -> int8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 4);
    let a: int8x16_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_s8<const LANE1: i32, const LANE2: i32>(
    a: int8x16_t,
    b: int8x16_t,
) -> int8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 4);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_s16<const LANE1: i32, const LANE2: i32>(
    a: int16x4_t,
    b: int16x8_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 3);
    let a: int16x8_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_s16<const LANE1: i32, const LANE2: i32>(
    a: int16x8_t,
    b: int16x8_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_s32<const LANE1: i32, const LANE2: i32>(
    a: int32x2_t,
    b: int32x4_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 2);
    let a: int32x4_t = simd_shuffle!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_s32<const LANE1: i32, const LANE2: i32>(
    a: int32x4_t,
    b: int32x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_s64<const LANE1: i32, const LANE2: i32>(
    a: int64x2_t,
    b: int64x2_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_u8<const LANE1: i32, const LANE2: i32>(
    a: uint8x8_t,
    b: uint8x16_t,
) -> uint8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 4);
    let a: uint8x16_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_u8<const LANE1: i32, const LANE2: i32>(
    a: uint8x16_t,
    b: uint8x16_t,
) -> uint8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 4);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_u16<const LANE1: i32, const LANE2: i32>(
    a: uint16x4_t,
    b: uint16x8_t,
) -> uint16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 3);
    let a: uint16x8_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_u16<const LANE1: i32, const LANE2: i32>(
    a: uint16x8_t,
    b: uint16x8_t,
) -> uint16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_u32<const LANE1: i32, const LANE2: i32>(
    a: uint32x2_t,
    b: uint32x4_t,
) -> uint32x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 2);
    let a: uint32x4_t = simd_shuffle!(a, a, [0, 1, 2, 3]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_u32<const LANE1: i32, const LANE2: i32>(
    a: uint32x4_t,
    b: uint32x4_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 2);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_u64<const LANE1: i32, const LANE2: i32>(
    a: uint64x2_t,
    b: uint64x2_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_p8<const LANE1: i32, const LANE2: i32>(
    a: poly8x8_t,
    b: poly8x16_t,
) -> poly8x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 4);
    let a: poly8x16_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [16 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 16 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 16 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 16 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 16 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 16 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 16 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 16 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_p8<const LANE1: i32, const LANE2: i32>(
    a: poly8x16_t,
    b: poly8x16_t,
) -> poly8x16_t {
    static_assert_uimm_bits!(LANE1, 4);
    static_assert_uimm_bits!(LANE2, 4);
    match LANE1 & 0b1111 {
        0 => simd_shuffle!(
            a,
            b,
            [
                16 + LANE2 as u32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        1 => simd_shuffle!(
            a,
            b,
            [
                0,
                16 + LANE2 as u32,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        2 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                16 + LANE2 as u32,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        3 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                16 + LANE2 as u32,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        4 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                16 + LANE2 as u32,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        5 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                16 + LANE2 as u32,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        6 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                16 + LANE2 as u32,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        7 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                16 + LANE2 as u32,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        8 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                16 + LANE2 as u32,
                9,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        9 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                16 + LANE2 as u32,
                10,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        10 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                16 + LANE2 as u32,
                11,
                12,
                13,
                14,
                15
            ]
        ),
        11 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                16 + LANE2 as u32,
                12,
                13,
                14,
                15
            ]
        ),
        12 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16 + LANE2 as u32,
                13,
                14,
                15
            ]
        ),
        13 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                16 + LANE2 as u32,
                14,
                15
            ]
        ),
        14 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                16 + LANE2 as u32,
                15
            ]
        ),
        15 => simd_shuffle!(
            a,
            b,
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                16 + LANE2 as u32
            ]
        ),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopy_laneq_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopy_laneq_p16<const LANE1: i32, const LANE2: i32>(
    a: poly16x4_t,
    b: poly16x8_t,
) -> poly16x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 3);
    let a: poly16x8_t = simd_shuffle!(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_p16<const LANE1: i32, const LANE2: i32>(
    a: poly16x8_t,
    b: poly16x8_t,
) -> poly16x8_t {
    static_assert_uimm_bits!(LANE1, 3);
    static_assert_uimm_bits!(LANE2, 3);
    match LANE1 & 0b111 {
        0 => simd_shuffle!(a, b, [8 + LANE2 as u32, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle!(a, b, [0, 8 + LANE2 as u32, 2, 3, 4, 5, 6, 7]),
        2 => simd_shuffle!(a, b, [0, 1, 8 + LANE2 as u32, 3, 4, 5, 6, 7]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 8 + LANE2 as u32, 4, 5, 6, 7]),
        4 => simd_shuffle!(a, b, [0, 1, 2, 3, 8 + LANE2 as u32, 5, 6, 7]),
        5 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 8 + LANE2 as u32, 6, 7]),
        6 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 8 + LANE2 as u32, 7]),
        7 => simd_shuffle!(a, b, [0, 1, 2, 3, 4, 5, 6, 8 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_laneq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 0, LANE2 = 1))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_laneq_p64<const LANE1: i32, const LANE2: i32>(
    a: poly64x2_t,
    b: poly64x2_t,
) -> poly64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert_uimm_bits!(LANE2, 1);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_f32<const LANE1: i32, const LANE2: i32>(
    a: float32x4_t,
    b: float32x2_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE1, 2);
    static_assert_uimm_bits!(LANE2, 1);
    let b: float32x4_t = simd_shuffle!(b, b, [0, 1, 2, 3]);
    match LANE1 & 0b11 {
        0 => simd_shuffle!(a, b, [4 + LANE2 as u32, 1, 2, 3]),
        1 => simd_shuffle!(a, b, [0, 4 + LANE2 as u32, 2, 3]),
        2 => simd_shuffle!(a, b, [0, 1, 4 + LANE2 as u32, 3]),
        3 => simd_shuffle!(a, b, [0, 1, 2, 4 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_f64<const LANE1: i32, const LANE2: i32>(
    a: float64x2_t,
    b: float64x1_t,
) -> float64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert!(LANE2 == 0);
    let b: float64x2_t = simd_shuffle!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_s64<const LANE1: i32, const LANE2: i32>(
    a: int64x2_t,
    b: int64x1_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert!(LANE2 == 0);
    let b: int64x2_t = simd_shuffle!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_u64<const LANE1: i32, const LANE2: i32>(
    a: uint64x2_t,
    b: uint64x1_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert!(LANE2 == 0);
    let b: uint64x2_t = simd_shuffle!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcopyq_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(mov, LANE1 = 1, LANE2 = 0))]
#[rustc_legacy_const_generics(1, 3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcopyq_lane_p64<const LANE1: i32, const LANE2: i32>(
    a: poly64x2_t,
    b: poly64x1_t,
) -> poly64x2_t {
    static_assert_uimm_bits!(LANE1, 1);
    static_assert!(LANE2 == 0);
    let b: poly64x2_t = simd_shuffle!(b, b, [0, 1]);
    match LANE1 & 0b1 {
        0 => simd_shuffle!(a, b, [2 + LANE2 as u32, 1]),
        1 => simd_shuffle!(a, b, [0, 2 + LANE2 as u32]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcreate_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcreate_f64(a: u64) -> float64x1_t {
    transmute(a)
}

#[doc = "Floating-point convert to lower precision narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_f32_f64(a: float64x2_t) -> float32x2_t {
    simd_cast(a)
}

#[doc = "Floating-point convert to higher precision long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_f64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_f64_f32(a: float32x2_t) -> float64x2_t {
    simd_cast(a)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_f64_s64(a: int64x1_t) -> float64x1_t {
    simd_cast(a)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_f64_s64(a: int64x2_t) -> float64x2_t {
    simd_cast(a)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_f64_u64(a: uint64x1_t) -> float64x1_t {
    simd_cast(a)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_f64_u64(a: uint64x2_t) -> float64x2_t {
    simd_cast(a)
}

#[doc = "Floating-point convert to lower precision narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_high_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_high_f32_f64(a: float32x2_t, b: float64x2_t) -> float32x4_t {
    simd_shuffle!(a, simd_cast(b), [0, 1, 2, 3])
}

#[doc = "Floating-point convert to higher precision long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_high_f64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_high_f64_f32(a: float32x4_t) -> float64x2_t {
    let b: float32x2_t = simd_shuffle!(a, a, [2, 3]);
    simd_cast(b)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_n_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_n_f64_s64<const N: i32>(a: int64x1_t) -> float64x1_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxs2fp.v1f64.v1i64"
        )]
        fn _vcvt_n_f64_s64(a: int64x1_t, n: i32) -> float64x1_t;
    }
    _vcvt_n_f64_s64(a, N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_n_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_n_f64_s64<const N: i32>(a: int64x2_t) -> float64x2_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxs2fp.v2f64.v2i64"
        )]
        fn _vcvtq_n_f64_s64(a: int64x2_t, n: i32) -> float64x2_t;
    }
    _vcvtq_n_f64_s64(a, N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_n_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_n_f64_u64<const N: i32>(a: uint64x1_t) -> float64x1_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxu2fp.v1f64.v1i64"
        )]
        fn _vcvt_n_f64_u64(a: int64x1_t, n: i32) -> float64x1_t;
    }
    _vcvt_n_f64_u64(a.as_signed(), N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_n_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_n_f64_u64<const N: i32>(a: uint64x2_t) -> float64x2_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxu2fp.v2f64.v2i64"
        )]
        fn _vcvtq_n_f64_u64(a: int64x2_t, n: i32) -> float64x2_t;
    }
    _vcvtq_n_f64_u64(a.as_signed(), N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_n_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_n_s64_f64<const N: i32>(a: float64x1_t) -> int64x1_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxs.v1i64.v1f64"
        )]
        fn _vcvt_n_s64_f64(a: float64x1_t, n: i32) -> int64x1_t;
    }
    _vcvt_n_s64_f64(a, N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_n_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_n_s64_f64<const N: i32>(a: float64x2_t) -> int64x2_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxs.v2i64.v2f64"
        )]
        fn _vcvtq_n_s64_f64(a: float64x2_t, n: i32) -> int64x2_t;
    }
    _vcvtq_n_s64_f64(a, N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_n_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_n_u64_f64<const N: i32>(a: float64x1_t) -> uint64x1_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxu.v1i64.v1f64"
        )]
        fn _vcvt_n_u64_f64(a: float64x1_t, n: i32) -> int64x1_t;
    }
    _vcvt_n_u64_f64(a, N).as_unsigned()
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_n_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_n_u64_f64<const N: i32>(a: float64x2_t) -> uint64x2_t {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxu.v2i64.v2f64"
        )]
        fn _vcvtq_n_u64_f64(a: float64x2_t, n: i32) -> int64x2_t;
    }
    _vcvtq_n_u64_f64(a, N).as_unsigned()
}

#[doc = "Floating-point convert to signed fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_s64_f64(a: float64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fptosi.sat.v1i64.v1f64"
        )]
        fn _vcvt_s64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvt_s64_f64(a)
}

#[doc = "Floating-point convert to signed fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_s64_f64(a: float64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fptosi.sat.v2i64.v2f64"
        )]
        fn _vcvtq_s64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtq_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvt_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvt_u64_f64(a: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fptoui.sat.v1i64.v1f64"
        )]
        fn _vcvt_u64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvt_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtq_u64_f64(a: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fptoui.sat.v2i64.v2f64"
        )]
        fn _vcvtq_u64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtq_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvta_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvta_s32_f32(a: float32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.v2i32.v2f32"
        )]
        fn _vcvta_s32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvta_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtaq_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtaq_s32_f32(a: float32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.v4i32.v4f32"
        )]
        fn _vcvtaq_s32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtaq_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvta_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvta_s64_f64(a: float64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.v1i64.v1f64"
        )]
        fn _vcvta_s64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvta_s64_f64(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtaq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtaq_s64_f64(a: float64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.v2i64.v2f64"
        )]
        fn _vcvtaq_s64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtaq_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvta_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvta_u32_f32(a: float32x2_t) -> uint32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.v2i32.v2f32"
        )]
        fn _vcvta_u32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvta_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtaq_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtaq_u32_f32(a: float32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.v4i32.v4f32"
        )]
        fn _vcvtaq_u32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtaq_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvta_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvta_u64_f64(a: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.v1i64.v1f64"
        )]
        fn _vcvta_u64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvta_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtaq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtaq_u64_f64(a: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.v2i64.v2f64"
        )]
        fn _vcvtaq_u64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtaq_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtas_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtas_s32_f32(a: f32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.i32.f32"
        )]
        fn _vcvtas_s32_f32(a: f32) -> i32;
    }
    _vcvtas_s32_f32(a)
}

#[doc = "Floating-point convert to integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtad_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtas))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtad_s64_f64(a: f64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtas.i64.f64"
        )]
        fn _vcvtad_s64_f64(a: f64) -> i64;
    }
    _vcvtad_s64_f64(a)
}

#[doc = "Floating-point convert to integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtas_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtas_u32_f32(a: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.i32.f32"
        )]
        fn _vcvtas_u32_f32(a: f32) -> i32;
    }
    _vcvtas_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to integer, rounding to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtad_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtau))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtad_u64_f64(a: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtau.i64.f64"
        )]
        fn _vcvtad_u64_f64(a: f64) -> i64;
    }
    _vcvtad_u64_f64(a).as_unsigned()
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_f64_s64(a: i64) -> f64 {
    a as f64
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_f32_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_f32_s32(a: i32) -> f32 {
    a as f32
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtm_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtm_s32_f32(a: float32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.v2i32.v2f32"
        )]
        fn _vcvtm_s32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtm_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmq_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmq_s32_f32(a: float32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.v4i32.v4f32"
        )]
        fn _vcvtmq_s32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtmq_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtm_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtm_s64_f64(a: float64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.v1i64.v1f64"
        )]
        fn _vcvtm_s64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtm_s64_f64(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmq_s64_f64(a: float64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.v2i64.v2f64"
        )]
        fn _vcvtmq_s64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtmq_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtm_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtm_u32_f32(a: float32x2_t) -> uint32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.v2i32.v2f32"
        )]
        fn _vcvtm_u32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtm_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmq_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmq_u32_f32(a: float32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.v4i32.v4f32"
        )]
        fn _vcvtmq_u32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtmq_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtm_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtm_u64_f64(a: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.v1i64.v1f64"
        )]
        fn _vcvtm_u64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtm_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmq_u64_f64(a: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.v2i64.v2f64"
        )]
        fn _vcvtmq_u64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtmq_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtms_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtms_s32_f32(a: f32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.i32.f32"
        )]
        fn _vcvtms_s32_f32(a: f32) -> i32;
    }
    _vcvtms_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmd_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtms))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmd_s64_f64(a: f64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtms.i64.f64"
        )]
        fn _vcvtmd_s64_f64(a: f64) -> i64;
    }
    _vcvtmd_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtms_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtms_u32_f32(a: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.i32.f32"
        )]
        fn _vcvtms_u32_f32(a: f32) -> i32;
    }
    _vcvtms_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtmd_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtmu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtmd_u64_f64(a: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtmu.i64.f64"
        )]
        fn _vcvtmd_u64_f64(a: f64) -> i64;
    }
    _vcvtmd_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtn_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtn_s32_f32(a: float32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.v2i32.v2f32"
        )]
        fn _vcvtn_s32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtn_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnq_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnq_s32_f32(a: float32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.v4i32.v4f32"
        )]
        fn _vcvtnq_s32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtnq_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtn_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtn_s64_f64(a: float64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.v1i64.v1f64"
        )]
        fn _vcvtn_s64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtn_s64_f64(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnq_s64_f64(a: float64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.v2i64.v2f64"
        )]
        fn _vcvtnq_s64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtnq_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtn_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtn_u32_f32(a: float32x2_t) -> uint32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.v2i32.v2f32"
        )]
        fn _vcvtn_u32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtn_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnq_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnq_u32_f32(a: float32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.v4i32.v4f32"
        )]
        fn _vcvtnq_u32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtnq_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtn_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtn_u64_f64(a: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.v1i64.v1f64"
        )]
        fn _vcvtn_u64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtn_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnq_u64_f64(a: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.v2i64.v2f64"
        )]
        fn _vcvtnq_u64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtnq_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtns_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtns_s32_f32(a: f32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.i32.f32"
        )]
        fn _vcvtns_s32_f32(a: f32) -> i32;
    }
    _vcvtns_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnd_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtns))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnd_s64_f64(a: f64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtns.i64.f64"
        )]
        fn _vcvtnd_s64_f64(a: f64) -> i64;
    }
    _vcvtnd_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtns_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtns_u32_f32(a: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.i32.f32"
        )]
        fn _vcvtns_u32_f32(a: f32) -> i32;
    }
    _vcvtns_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtnd_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtnu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtnd_u64_f64(a: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtnu.i64.f64"
        )]
        fn _vcvtnd_u64_f64(a: f64) -> i64;
    }
    _vcvtnd_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtp_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtp_s32_f32(a: float32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.v2i32.v2f32"
        )]
        fn _vcvtp_s32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtp_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpq_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpq_s32_f32(a: float32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.v4i32.v4f32"
        )]
        fn _vcvtpq_s32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtpq_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtp_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtp_s64_f64(a: float64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.v1i64.v1f64"
        )]
        fn _vcvtp_s64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtp_s64_f64(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpq_s64_f64(a: float64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.v2i64.v2f64"
        )]
        fn _vcvtpq_s64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtpq_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtp_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtp_u32_f32(a: float32x2_t) -> uint32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.v2i32.v2f32"
        )]
        fn _vcvtp_u32_f32(a: float32x2_t) -> int32x2_t;
    }
    _vcvtp_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpq_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpq_u32_f32(a: float32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.v4i32.v4f32"
        )]
        fn _vcvtpq_u32_f32(a: float32x4_t) -> int32x4_t;
    }
    _vcvtpq_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtp_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtp_u64_f64(a: float64x1_t) -> uint64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.v1i64.v1f64"
        )]
        fn _vcvtp_u64_f64(a: float64x1_t) -> int64x1_t;
    }
    _vcvtp_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpq_u64_f64(a: float64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.v2i64.v2f64"
        )]
        fn _vcvtpq_u64_f64(a: float64x2_t) -> int64x2_t;
    }
    _vcvtpq_u64_f64(a).as_unsigned()
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtps_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtps_s32_f32(a: f32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.i32.f32"
        )]
        fn _vcvtps_s32_f32(a: f32) -> i32;
    }
    _vcvtps_s32_f32(a)
}

#[doc = "Floating-point convert to signed integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpd_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpd_s64_f64(a: f64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtps.i64.f64"
        )]
        fn _vcvtpd_s64_f64(a: f64) -> i64;
    }
    _vcvtpd_s64_f64(a)
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtps_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtps_u32_f32(a: f32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.i32.f32"
        )]
        fn _vcvtps_u32_f32(a: f32) -> i32;
    }
    _vcvtps_u32_f32(a).as_unsigned()
}

#[doc = "Floating-point convert to unsigned integer, rounding toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtpd_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtpu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtpd_u64_f64(a: f64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtpu.i64.f64"
        )]
        fn _vcvtpd_u64_f64(a: f64) -> i64;
    }
    _vcvtpd_u64_f64(a).as_unsigned()
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_f32_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_f32_u32(a: u32) -> f32 {
    a as f32
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_f64_u64(a: u64) -> f64 {
    a as f64
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_n_f32_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_n_f32_s32<const N: i32>(a: i32) -> f32 {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxs2fp.f32.i32"
        )]
        fn _vcvts_n_f32_s32(a: i32, n: i32) -> f32;
    }
    _vcvts_n_f32_s32(a, N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_n_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_n_f64_s64<const N: i32>(a: i64) -> f64 {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxs2fp.f64.i64"
        )]
        fn _vcvtd_n_f64_s64(a: i64, n: i32) -> f64;
    }
    _vcvtd_n_f64_s64(a, N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_n_f32_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_n_f32_u32<const N: i32>(a: u32) -> f32 {
    static_assert!(N >= 1 && N <= 32);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxu2fp.f32.i32"
        )]
        fn _vcvts_n_f32_u32(a: i32, n: i32) -> f32;
    }
    _vcvts_n_f32_u32(a.as_signed(), N)
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_n_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_n_f64_u64<const N: i32>(a: u64) -> f64 {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfxu2fp.f64.i64"
        )]
        fn _vcvtd_n_f64_u64(a: i64, n: i32) -> f64;
    }
    _vcvtd_n_f64_u64(a.as_signed(), N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_n_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_n_s32_f32<const N: i32>(a: f32) -> i32 {
    static_assert!(N >= 1 && N <= 32);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxs.i32.f32"
        )]
        fn _vcvts_n_s32_f32(a: f32, n: i32) -> i32;
    }
    _vcvts_n_s32_f32(a, N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_n_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_n_s64_f64<const N: i32>(a: f64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxs.i64.f64"
        )]
        fn _vcvtd_n_s64_f64(a: f64, n: i32) -> i64;
    }
    _vcvtd_n_s64_f64(a, N)
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_n_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_n_u32_f32<const N: i32>(a: f32) -> u32 {
    static_assert!(N >= 1 && N <= 32);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxu.i32.f32"
        )]
        fn _vcvts_n_u32_f32(a: f32, n: i32) -> i32;
    }
    _vcvts_n_u32_f32(a, N).as_unsigned()
}

#[doc = "Floating-point convert to fixed-point, rounding toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_n_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_n_u64_f64<const N: i32>(a: f64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.vcvtfp2fxu.i64.f64"
        )]
        fn _vcvtd_n_u64_f64(a: f64, n: i32) -> i64;
    }
    _vcvtd_n_u64_f64(a, N).as_unsigned()
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_s32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_s32_f32(a: f32) -> i32 {
    a as i32
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzs))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_s64_f64(a: f64) -> i64 {
    a as i64
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvts_u32_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvts_u32_f32(a: f32) -> u32 {
    a as u32
}

#[doc = "Fixed-point convert to floating-point"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtd_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtzu))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtd_u64_f64(a: f64) -> u64 {
    a as u64
}

#[doc = "Floating-point convert to lower precision narrow, rounding to odd"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtx_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtxn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtx_f32_f64(a: float64x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fcvtxn.v2f32.v2f64"
        )]
        fn _vcvtx_f32_f64(a: float64x2_t) -> float32x2_t;
    }
    _vcvtx_f32_f64(a)
}

#[doc = "Floating-point convert to lower precision narrow, rounding to odd"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtx_high_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtxn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtx_high_f32_f64(a: float32x2_t, b: float64x2_t) -> float32x4_t {
    simd_shuffle!(a, vcvtx_f32_f64(b), [0, 1, 2, 3])
}

#[doc = "Floating-point convert to lower precision narrow, rounding to odd"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vcvtxd_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fcvtxn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vcvtxd_f32_f64(a: f64) -> f32 {
    simd_extract!(vcvtx_f32_f64(vdupq_n_f64(a)), 0)
}

#[doc = "Divide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdiv_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdiv_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_div(a, b)
}

#[doc = "Divide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdivq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdivq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_div(a, b)
}

#[doc = "Divide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdiv_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdiv_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_div(a, b)
}

#[doc = "Divide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdivq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fdiv))]
pub unsafe fn vdivq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_div(a, b)
}

#[doc = "Dot product arithmetic (indexed)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdot_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[cfg_attr(test, assert_instr(sdot, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_dotprod", issue = "117224")]
pub unsafe fn vdot_laneq_s32<const LANE: i32>(
    a: int32x2_t,
    b: int8x8_t,
    c: int8x16_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = transmute(c);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vdot_s32(a, b, transmute(c))
}

#[doc = "Dot product arithmetic (indexed)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdotq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[cfg_attr(test, assert_instr(sdot, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_dotprod", issue = "117224")]
pub unsafe fn vdotq_laneq_s32<const LANE: i32>(
    a: int32x4_t,
    b: int8x16_t,
    c: int8x16_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = transmute(c);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vdotq_s32(a, b, transmute(c))
}

#[doc = "Dot product arithmetic (indexed)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdot_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[cfg_attr(test, assert_instr(udot, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_dotprod", issue = "117224")]
pub unsafe fn vdot_laneq_u32<const LANE: i32>(
    a: uint32x2_t,
    b: uint8x8_t,
    c: uint8x16_t,
) -> uint32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: uint32x4_t = transmute(c);
    let c: uint32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vdot_u32(a, b, transmute(c))
}

#[doc = "Dot product arithmetic (indexed)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdotq_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,dotprod")]
#[cfg_attr(test, assert_instr(udot, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_dotprod", issue = "117224")]
pub unsafe fn vdotq_laneq_u32<const LANE: i32>(
    a: uint32x4_t,
    b: uint8x16_t,
    c: uint8x16_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: uint32x4_t = transmute(c);
    let c: uint32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vdotq_u32(a, b, transmute(c))
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdup_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdup_lane_f64<const N: i32>(a: float64x1_t) -> float64x1_t {
    static_assert!(N == 0);
    a
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdup_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdup_lane_p64<const N: i32>(a: poly64x1_t) -> poly64x1_t {
    static_assert!(N == 0);
    a
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdup_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdup_laneq_f64<const N: i32>(a: float64x2_t) -> float64x1_t {
    static_assert_uimm_bits!(N, 1);
    transmute::<f64, _>(simd_extract!(a, N as u32))
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdup_laneq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdup_laneq_p64<const N: i32>(a: poly64x2_t) -> poly64x1_t {
    static_assert_uimm_bits!(N, 1);
    transmute::<u64, _>(simd_extract!(a, N as u32))
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_lane_s8<const N: i32>(a: int8x8_t) -> i8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_laneq_s16<const N: i32>(a: int16x8_t) -> i16 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_lane_u8<const N: i32>(a: uint8x8_t) -> u8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_laneq_u16<const N: i32>(a: uint16x8_t) -> u16 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_lane_p8<const N: i32>(a: poly8x8_t) -> p8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_laneq_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 4))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_laneq_p16<const N: i32>(a: poly16x8_t) -> p16 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_laneq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_laneq_s8<const N: i32>(a: int8x16_t) -> i8 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_laneq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_laneq_u8<const N: i32>(a: uint8x16_t) -> u8 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupb_laneq_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 8))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupb_laneq_p8<const N: i32>(a: poly8x16_t) -> p8 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_lane_f64<const N: i32>(a: float64x1_t) -> f64 {
    static_assert!(N == 0);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_lane_s64<const N: i32>(a: int64x1_t) -> i64 {
    static_assert!(N == 0);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_lane_u64<const N: i32>(a: uint64x1_t) -> u64 {
    static_assert!(N == 0);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupq_lane_f64<const N: i32>(a: float64x1_t) -> float64x2_t {
    static_assert!(N == 0);
    simd_shuffle!(a, a, [N as u32, N as u32])
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupq_lane_p64<const N: i32>(a: poly64x1_t) -> poly64x2_t {
    static_assert!(N == 0);
    simd_shuffle!(a, a, [N as u32, N as u32])
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupq_laneq_f64<const N: i32>(a: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(N, 1);
    simd_shuffle!(a, a, [N as u32, N as u32])
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupq_laneq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupq_laneq_p64<const N: i32>(a: poly64x2_t) -> poly64x2_t {
    static_assert_uimm_bits!(N, 1);
    simd_shuffle!(a, a, [N as u32, N as u32])
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_lane_f32<const N: i32>(a: float32x2_t) -> f32 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_laneq_f64<const N: i32>(a: float64x2_t) -> f64 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_lane_s32<const N: i32>(a: int32x2_t) -> i32 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_laneq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_laneq_s64<const N: i32>(a: int64x2_t) -> i64 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_lane_u32<const N: i32>(a: uint32x2_t) -> u32 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdupd_laneq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdupd_laneq_u64<const N: i32>(a: uint64x2_t) -> u64 {
    static_assert_uimm_bits!(N, 1);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_laneq_f32<const N: i32>(a: float32x4_t) -> f32 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_lane_s16<const N: i32>(a: int16x4_t) -> i16 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_laneq_s32<const N: i32>(a: int32x4_t) -> i32 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_lane_u16<const N: i32>(a: uint16x4_t) -> u16 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vdups_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vdups_laneq_u32<const N: i32>(a: uint32x4_t) -> u32 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Set all vector lanes to the same value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vduph_lane_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vduph_lane_p16<const N: i32>(a: poly16x4_t) -> p16 {
    static_assert_uimm_bits!(N, 2);
    simd_extract!(a, N as u32)
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3s.v16i8"
        )]
        fn _veor3q_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
    }
    _veor3q_s8(a, b, c)
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3s.v8i16"
        )]
        fn _veor3q_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _veor3q_s16(a, b, c)
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3s.v4i32"
        )]
        fn _veor3q_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _veor3q_s32(a, b, c)
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3s.v2i64"
        )]
        fn _veor3q_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _veor3q_s64(a, b, c)
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3u.v16i8"
        )]
        fn _veor3q_u8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t;
    }
    _veor3q_u8(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_u16(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3u.v8i16"
        )]
        fn _veor3q_u16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _veor3q_u16(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3u.v4i32"
        )]
        fn _veor3q_u32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _veor3q_u32(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Three-way exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/veor3q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
#[cfg_attr(test, assert_instr(eor3))]
pub unsafe fn veor3q_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.eor3u.v2i64"
        )]
        fn _veor3q_u64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _veor3q_u64(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Extract vector from pair of vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vextq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vextq_f64<const N: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(N, 1);
    match N & 0b1 {
        0 => simd_shuffle!(a, b, [0, 1]),
        1 => simd_shuffle!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Extract vector from pair of vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vextq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vextq_p64<const N: i32>(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    static_assert_uimm_bits!(N, 1);
    match N & 0b1 {
        0 => simd_shuffle!(a, b, [0, 1]),
        1 => simd_shuffle!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

#[doc = "Floating-point fused Multiply-Add to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmadd))]
pub unsafe fn vfma_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.v1f64"
        )]
        fn _vfma_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t;
    }
    _vfma_f64(b, c, a)
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfma_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vfma_f32(a, b, vdup_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfma_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vfma_f32(a, b, vdup_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmaq_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    vfmaq_f32(a, b, vdupq_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmaq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vfmaq_f32(a, b, vdupq_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmaq_laneq_f64<const LANE: i32>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x2_t,
) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vfmaq_f64(a, b, vdupq_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfma_lane_f64<const LANE: i32>(
    a: float64x1_t,
    b: float64x1_t,
    c: float64x1_t,
) -> float64x1_t {
    static_assert!(LANE == 0);
    vfma_f64(a, b, vdup_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfma_laneq_f64<const LANE: i32>(
    a: float64x1_t,
    b: float64x1_t,
    c: float64x2_t,
) -> float64x1_t {
    static_assert_uimm_bits!(LANE, 1);
    vfma_f64(a, b, vdup_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused Multiply-Add to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfma_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmadd))]
pub unsafe fn vfma_n_f64(a: float64x1_t, b: float64x1_t, c: f64) -> float64x1_t {
    vfma_f64(a, b, vdup_n_f64(c))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmad_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmad_lane_f64<const LANE: i32>(a: f64, b: f64, c: float64x1_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.f64"
        )]
        fn _vfmad_lane_f64(a: f64, b: f64, c: f64) -> f64;
    }
    static_assert!(LANE == 0);
    let c: f64 = simd_extract!(c, LANE as u32);
    _vfmad_lane_f64(b, c, a)
}

#[doc = "Floating-point fused Multiply-Add to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmla))]
pub unsafe fn vfmaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.v2f64"
        )]
        fn _vfmaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t;
    }
    _vfmaq_f64(b, c, a)
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmla, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmaq_lane_f64<const LANE: i32>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x1_t,
) -> float64x2_t {
    static_assert!(LANE == 0);
    vfmaq_f64(a, b, vdupq_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused Multiply-Add to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmaq_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmla))]
pub unsafe fn vfmaq_n_f64(a: float64x2_t, b: float64x2_t, c: f64) -> float64x2_t {
    vfmaq_f64(a, b, vdupq_n_f64(c))
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmas_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmas_lane_f32<const LANE: i32>(a: f32, b: f32, c: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.f32"
        )]
        fn _vfmas_lane_f32(a: f32, b: f32, c: f32) -> f32;
    }
    static_assert_uimm_bits!(LANE, 1);
    let c: f32 = simd_extract!(c, LANE as u32);
    _vfmas_lane_f32(b, c, a)
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmas_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmas_laneq_f32<const LANE: i32>(a: f32, b: f32, c: float32x4_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.f32"
        )]
        fn _vfmas_laneq_f32(a: f32, b: f32, c: f32) -> f32;
    }
    static_assert_uimm_bits!(LANE, 2);
    let c: f32 = simd_extract!(c, LANE as u32);
    _vfmas_laneq_f32(b, c, a)
}

#[doc = "Floating-point fused multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmad_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmadd, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmad_laneq_f64<const LANE: i32>(a: f64, b: f64, c: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.fma.f64"
        )]
        fn _vfmad_laneq_f64(a: f64, b: f64, c: f64) -> f64;
    }
    static_assert_uimm_bits!(LANE, 1);
    let c: f64 = simd_extract!(c, LANE as u32);
    _vfmad_laneq_f64(b, c, a)
}

#[doc = "Floating-point fused multiply-subtract from accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    let b: float64x1_t = simd_neg(b);
    vfma_f64(a, b, c)
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_lane_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x2_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vfms_f32(a, b, vdup_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_laneq_f32<const LANE: i32>(
    a: float32x2_t,
    b: float32x2_t,
    c: float32x4_t,
) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vfms_f32(a, b, vdup_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_lane_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x2_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    vfmsq_f32(a, b, vdupq_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_laneq_f32<const LANE: i32>(
    a: float32x4_t,
    b: float32x4_t,
    c: float32x4_t,
) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vfmsq_f32(a, b, vdupq_n_f32(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_laneq_f64<const LANE: i32>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x2_t,
) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vfmsq_f64(a, b, vdupq_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_lane_f64<const LANE: i32>(
    a: float64x1_t,
    b: float64x1_t,
    c: float64x1_t,
) -> float64x1_t {
    static_assert!(LANE == 0);
    vfms_f64(a, b, vdup_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_laneq_f64<const LANE: i32>(
    a: float64x1_t,
    b: float64x1_t,
    c: float64x2_t,
) -> float64x1_t {
    static_assert_uimm_bits!(LANE, 1);
    vfms_f64(a, b, vdup_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused Multiply-subtract to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfms_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfms_n_f64(a: float64x1_t, b: float64x1_t, c: f64) -> float64x1_t {
    vfms_f64(a, b, vdup_n_f64(c))
}

#[doc = "Floating-point fused multiply-subtract from accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    let b: float64x2_t = simd_neg(b);
    vfmaq_f64(a, b, c)
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_lane_f64<const LANE: i32>(
    a: float64x2_t,
    b: float64x2_t,
    c: float64x1_t,
) -> float64x2_t {
    static_assert!(LANE == 0);
    vfmsq_f64(a, b, vdupq_n_f64(simd_extract!(c, LANE as u32)))
}

#[doc = "Floating-point fused Multiply-subtract to accumulator(vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsq_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmls))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsq_n_f64(a: float64x2_t, b: float64x2_t, c: f64) -> float64x2_t {
    vfmsq_f64(a, b, vdupq_n_f64(c))
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmss_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmss_lane_f32<const LANE: i32>(a: f32, b: f32, c: float32x2_t) -> f32 {
    vfmas_lane_f32::<LANE>(a, -b, c)
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmss_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmss_laneq_f32<const LANE: i32>(a: f32, b: f32, c: float32x4_t) -> f32 {
    vfmas_laneq_f32::<LANE>(a, -b, c)
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsd_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsd_lane_f64<const LANE: i32>(a: f64, b: f64, c: float64x1_t) -> f64 {
    vfmad_lane_f64::<LANE>(a, -b, c)
}

#[doc = "Floating-point fused multiply-subtract to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vfmsd_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmsub, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vfmsd_laneq_f64<const LANE: i32>(a: f64, b: f64, c: float64x2_t) -> f64 {
    vfmad_laneq_f64::<LANE>(a, -b, c)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1_f64_x2)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1_f64_x2(a: *const f64) -> float64x1x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x2.v1f64.p0f64"
        )]
        fn _vld1_f64_x2(a: *const f64) -> float64x1x2_t;
    }
    _vld1_f64_x2(a)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1_f64_x3)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1_f64_x3(a: *const f64) -> float64x1x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x3.v1f64.p0f64"
        )]
        fn _vld1_f64_x3(a: *const f64) -> float64x1x3_t;
    }
    _vld1_f64_x3(a)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1_f64_x4)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1_f64_x4(a: *const f64) -> float64x1x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x4.v1f64.p0f64"
        )]
        fn _vld1_f64_x4(a: *const f64) -> float64x1x4_t;
    }
    _vld1_f64_x4(a)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1q_f64_x2)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1q_f64_x2(a: *const f64) -> float64x2x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x2.v2f64.p0f64"
        )]
        fn _vld1q_f64_x2(a: *const f64) -> float64x2x2_t;
    }
    _vld1q_f64_x2(a)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1q_f64_x3)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1q_f64_x3(a: *const f64) -> float64x2x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x3.v2f64.p0f64"
        )]
        fn _vld1q_f64_x3(a: *const f64) -> float64x2x3_t;
    }
    _vld1q_f64_x3(a)
}

#[doc = "Load multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld1q_f64_x4)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld1))]
pub unsafe fn vld1q_f64_x4(a: *const f64) -> float64x2x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld1x4.v2f64.p0f64"
        )]
        fn _vld1q_f64_x4(a: *const f64) -> float64x2x4_t;
    }
    _vld1q_f64_x4(a)
}

#[doc = "Load single 2-element structure and replicate to all lanes of two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2r))]
pub unsafe fn vld2_dup_f64(a: *const f64) -> float64x1x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2r.v1f64.p0f64"
        )]
        fn _vld2_dup_f64(ptr: *const f64) -> float64x1x2_t;
    }
    _vld2_dup_f64(a as _)
}

#[doc = "Load single 2-element structure and replicate to all lanes of two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2r))]
pub unsafe fn vld2q_dup_f64(a: *const f64) -> float64x2x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2r.v2f64.p0f64"
        )]
        fn _vld2q_dup_f64(ptr: *const f64) -> float64x2x2_t;
    }
    _vld2q_dup_f64(a as _)
}

#[doc = "Load single 2-element structure and replicate to all lanes of two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_dup_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2r))]
pub unsafe fn vld2q_dup_s64(a: *const i64) -> int64x2x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2r.v2i64.p0i64"
        )]
        fn _vld2q_dup_s64(ptr: *const i64) -> int64x2x2_t;
    }
    _vld2q_dup_s64(a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vld2_f64(a: *const f64) -> float64x1x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2.v1f64.p0v1f64"
        )]
        fn _vld2_f64(ptr: *const float64x1_t) -> float64x1x2_t;
    }
    _vld2_f64(a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2_lane_f64<const LANE: i32>(a: *const f64, b: float64x1x2_t) -> float64x1x2_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2lane.v1f64.p0i8"
        )]
        fn _vld2_lane_f64(a: float64x1_t, b: float64x1_t, n: i64, ptr: *const i8) -> float64x1x2_t;
    }
    _vld2_lane_f64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2_lane_s64<const LANE: i32>(a: *const i64, b: int64x1x2_t) -> int64x1x2_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2lane.v1i64.p0i8"
        )]
        fn _vld2_lane_s64(a: int64x1_t, b: int64x1_t, n: i64, ptr: *const i8) -> int64x1x2_t;
    }
    _vld2_lane_s64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2_lane_p64<const LANE: i32>(a: *const p64, b: poly64x1x2_t) -> poly64x1x2_t {
    static_assert!(LANE == 0);
    transmute(vld2_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2_lane_u64<const LANE: i32>(a: *const u64, b: uint64x1x2_t) -> uint64x1x2_t {
    static_assert!(LANE == 0);
    transmute(vld2_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load single 2-element structure and replicate to all lanes of two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_dup_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2r))]
pub unsafe fn vld2q_dup_p64(a: *const p64) -> poly64x2x2_t {
    transmute(vld2q_dup_s64(transmute(a)))
}

#[doc = "Load single 2-element structure and replicate to all lanes of two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_dup_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2r))]
pub unsafe fn vld2q_dup_u64(a: *const u64) -> uint64x2x2_t {
    transmute(vld2q_dup_s64(transmute(a)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2))]
pub unsafe fn vld2q_f64(a: *const f64) -> float64x2x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2.v2f64.p0v2f64"
        )]
        fn _vld2q_f64(ptr: *const float64x2_t) -> float64x2x2_t;
    }
    _vld2q_f64(a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2))]
pub unsafe fn vld2q_s64(a: *const i64) -> int64x2x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2.v2i64.p0v2i64"
        )]
        fn _vld2q_s64(ptr: *const int64x2_t) -> int64x2x2_t;
    }
    _vld2q_s64(a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_f64<const LANE: i32>(a: *const f64, b: float64x2x2_t) -> float64x2x2_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2lane.v2f64.p0i8"
        )]
        fn _vld2q_lane_f64(a: float64x2_t, b: float64x2_t, n: i64, ptr: *const i8)
            -> float64x2x2_t;
    }
    _vld2q_lane_f64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_s8<const LANE: i32>(a: *const i8, b: int8x16x2_t) -> int8x16x2_t {
    static_assert_uimm_bits!(LANE, 4);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2lane.v16i8.p0i8"
        )]
        fn _vld2q_lane_s8(a: int8x16_t, b: int8x16_t, n: i64, ptr: *const i8) -> int8x16x2_t;
    }
    _vld2q_lane_s8(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_s64<const LANE: i32>(a: *const i64, b: int64x2x2_t) -> int64x2x2_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld2lane.v2i64.p0i8"
        )]
        fn _vld2q_lane_s64(a: int64x2_t, b: int64x2_t, n: i64, ptr: *const i8) -> int64x2x2_t;
    }
    _vld2q_lane_s64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_p64<const LANE: i32>(a: *const p64, b: poly64x2x2_t) -> poly64x2x2_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld2q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_u8<const LANE: i32>(a: *const u8, b: uint8x16x2_t) -> uint8x16x2_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld2q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_u64<const LANE: i32>(a: *const u64, b: uint64x2x2_t) -> uint64x2x2_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld2q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld2q_lane_p8<const LANE: i32>(a: *const p8, b: poly8x16x2_t) -> poly8x16x2_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld2q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2))]
pub unsafe fn vld2q_p64(a: *const p64) -> poly64x2x2_t {
    transmute(vld2q_s64(transmute(a)))
}

#[doc = "Load multiple 2-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld2))]
pub unsafe fn vld2q_u64(a: *const u64) -> uint64x2x2_t {
    transmute(vld2q_s64(transmute(a)))
}

#[doc = "Load single 3-element structure and replicate to all lanes of three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3r))]
pub unsafe fn vld3_dup_f64(a: *const f64) -> float64x1x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3r.v1f64.p0f64"
        )]
        fn _vld3_dup_f64(ptr: *const f64) -> float64x1x3_t;
    }
    _vld3_dup_f64(a as _)
}

#[doc = "Load single 3-element structure and replicate to all lanes of three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3r))]
pub unsafe fn vld3q_dup_f64(a: *const f64) -> float64x2x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3r.v2f64.p0f64"
        )]
        fn _vld3q_dup_f64(ptr: *const f64) -> float64x2x3_t;
    }
    _vld3q_dup_f64(a as _)
}

#[doc = "Load single 3-element structure and replicate to all lanes of three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_dup_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3r))]
pub unsafe fn vld3q_dup_s64(a: *const i64) -> int64x2x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3r.v2i64.p0i64"
        )]
        fn _vld3q_dup_s64(ptr: *const i64) -> int64x2x3_t;
    }
    _vld3q_dup_s64(a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vld3_f64(a: *const f64) -> float64x1x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3.v1f64.p0v1f64"
        )]
        fn _vld3_f64(ptr: *const float64x1_t) -> float64x1x3_t;
    }
    _vld3_f64(a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3_lane_f64<const LANE: i32>(a: *const f64, b: float64x1x3_t) -> float64x1x3_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3lane.v1f64.p0i8"
        )]
        fn _vld3_lane_f64(
            a: float64x1_t,
            b: float64x1_t,
            c: float64x1_t,
            n: i64,
            ptr: *const i8,
        ) -> float64x1x3_t;
    }
    _vld3_lane_f64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3_lane_p64<const LANE: i32>(a: *const p64, b: poly64x1x3_t) -> poly64x1x3_t {
    static_assert!(LANE == 0);
    transmute(vld3_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 3-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3_lane_s64<const LANE: i32>(a: *const i64, b: int64x1x3_t) -> int64x1x3_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3lane.v1i64.p0i8"
        )]
        fn _vld3_lane_s64(
            a: int64x1_t,
            b: int64x1_t,
            c: int64x1_t,
            n: i64,
            ptr: *const i8,
        ) -> int64x1x3_t;
    }
    _vld3_lane_s64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3_lane_u64<const LANE: i32>(a: *const u64, b: uint64x1x3_t) -> uint64x1x3_t {
    static_assert!(LANE == 0);
    transmute(vld3_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load single 3-element structure and replicate to all lanes of three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_dup_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3r))]
pub unsafe fn vld3q_dup_p64(a: *const p64) -> poly64x2x3_t {
    transmute(vld3q_dup_s64(transmute(a)))
}

#[doc = "Load single 3-element structure and replicate to all lanes of three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_dup_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3r))]
pub unsafe fn vld3q_dup_u64(a: *const u64) -> uint64x2x3_t {
    transmute(vld3q_dup_s64(transmute(a)))
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3))]
pub unsafe fn vld3q_f64(a: *const f64) -> float64x2x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3.v2f64.p0v2f64"
        )]
        fn _vld3q_f64(ptr: *const float64x2_t) -> float64x2x3_t;
    }
    _vld3q_f64(a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3))]
pub unsafe fn vld3q_s64(a: *const i64) -> int64x2x3_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3.v2i64.p0v2i64"
        )]
        fn _vld3q_s64(ptr: *const int64x2_t) -> int64x2x3_t;
    }
    _vld3q_s64(a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_f64<const LANE: i32>(a: *const f64, b: float64x2x3_t) -> float64x2x3_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3lane.v2f64.p0i8"
        )]
        fn _vld3q_lane_f64(
            a: float64x2_t,
            b: float64x2_t,
            c: float64x2_t,
            n: i64,
            ptr: *const i8,
        ) -> float64x2x3_t;
    }
    _vld3q_lane_f64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_p64<const LANE: i32>(a: *const p64, b: poly64x2x3_t) -> poly64x2x3_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld3q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 3-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_s8<const LANE: i32>(a: *const i8, b: int8x16x3_t) -> int8x16x3_t {
    static_assert_uimm_bits!(LANE, 3);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3lane.v16i8.p0i8"
        )]
        fn _vld3q_lane_s8(
            a: int8x16_t,
            b: int8x16_t,
            c: int8x16_t,
            n: i64,
            ptr: *const i8,
        ) -> int8x16x3_t;
    }
    _vld3q_lane_s8(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Load multiple 3-element structures to two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_s64<const LANE: i32>(a: *const i64, b: int64x2x3_t) -> int64x2x3_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld3lane.v2i64.p0i8"
        )]
        fn _vld3q_lane_s64(
            a: int64x2_t,
            b: int64x2_t,
            c: int64x2_t,
            n: i64,
            ptr: *const i8,
        ) -> int64x2x3_t;
    }
    _vld3q_lane_s64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_u8<const LANE: i32>(a: *const u8, b: uint8x16x3_t) -> uint8x16x3_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld3q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_u64<const LANE: i32>(a: *const u64, b: uint64x2x3_t) -> uint64x2x3_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld3q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld3q_lane_p8<const LANE: i32>(a: *const p8, b: poly8x16x3_t) -> poly8x16x3_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld3q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3))]
pub unsafe fn vld3q_p64(a: *const p64) -> poly64x2x3_t {
    transmute(vld3q_s64(transmute(a)))
}

#[doc = "Load multiple 3-element structures to three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld3q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld3))]
pub unsafe fn vld3q_u64(a: *const u64) -> uint64x2x3_t {
    transmute(vld3q_s64(transmute(a)))
}

#[doc = "Load single 4-element structure and replicate to all lanes of four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4_dup_f64(a: *const f64) -> float64x1x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4r.v1f64.p0f64"
        )]
        fn _vld4_dup_f64(ptr: *const f64) -> float64x1x4_t;
    }
    _vld4_dup_f64(a as _)
}

#[doc = "Load single 4-element structure and replicate to all lanes of four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_dup_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_dup_f64(a: *const f64) -> float64x2x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4r.v2f64.p0f64"
        )]
        fn _vld4q_dup_f64(ptr: *const f64) -> float64x2x4_t;
    }
    _vld4q_dup_f64(a as _)
}

#[doc = "Load single 4-element structure and replicate to all lanes of four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_dup_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_dup_s64(a: *const i64) -> int64x2x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4r.v2i64.p0i64"
        )]
        fn _vld4q_dup_s64(ptr: *const i64) -> int64x2x4_t;
    }
    _vld4q_dup_s64(a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vld4_f64(a: *const f64) -> float64x1x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4.v1f64.p0v1f64"
        )]
        fn _vld4_f64(ptr: *const float64x1_t) -> float64x1x4_t;
    }
    _vld4_f64(a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4_lane_f64<const LANE: i32>(a: *const f64, b: float64x1x4_t) -> float64x1x4_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4lane.v1f64.p0i8"
        )]
        fn _vld4_lane_f64(
            a: float64x1_t,
            b: float64x1_t,
            c: float64x1_t,
            d: float64x1_t,
            n: i64,
            ptr: *const i8,
        ) -> float64x1x4_t;
    }
    _vld4_lane_f64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4_lane_s64<const LANE: i32>(a: *const i64, b: int64x1x4_t) -> int64x1x4_t {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4lane.v1i64.p0i8"
        )]
        fn _vld4_lane_s64(
            a: int64x1_t,
            b: int64x1_t,
            c: int64x1_t,
            d: int64x1_t,
            n: i64,
            ptr: *const i8,
        ) -> int64x1x4_t;
    }
    _vld4_lane_s64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4_lane_p64<const LANE: i32>(a: *const p64, b: poly64x1x4_t) -> poly64x1x4_t {
    static_assert!(LANE == 0);
    transmute(vld4_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4_lane_u64<const LANE: i32>(a: *const u64, b: uint64x1x4_t) -> uint64x1x4_t {
    static_assert!(LANE == 0);
    transmute(vld4_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load single 4-element structure and replicate to all lanes of four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_dup_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld4r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_dup_p64(a: *const p64) -> poly64x2x4_t {
    transmute(vld4q_dup_s64(transmute(a)))
}

#[doc = "Load single 4-element structure and replicate to all lanes of four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_dup_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4r))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_dup_u64(a: *const u64) -> uint64x2x4_t {
    transmute(vld4q_dup_s64(transmute(a)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld4))]
pub unsafe fn vld4q_f64(a: *const f64) -> float64x2x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4.v2f64.p0v2f64"
        )]
        fn _vld4q_f64(ptr: *const float64x2_t) -> float64x2x4_t;
    }
    _vld4q_f64(a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld4))]
pub unsafe fn vld4q_s64(a: *const i64) -> int64x2x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4.v2i64.p0v2i64"
        )]
        fn _vld4q_s64(ptr: *const int64x2_t) -> int64x2x4_t;
    }
    _vld4q_s64(a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_f64<const LANE: i32>(a: *const f64, b: float64x2x4_t) -> float64x2x4_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4lane.v2f64.p0i8"
        )]
        fn _vld4q_lane_f64(
            a: float64x2_t,
            b: float64x2_t,
            c: float64x2_t,
            d: float64x2_t,
            n: i64,
            ptr: *const i8,
        ) -> float64x2x4_t;
    }
    _vld4q_lane_f64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_s8<const LANE: i32>(a: *const i8, b: int8x16x4_t) -> int8x16x4_t {
    static_assert_uimm_bits!(LANE, 3);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4lane.v16i8.p0i8"
        )]
        fn _vld4q_lane_s8(
            a: int8x16_t,
            b: int8x16_t,
            c: int8x16_t,
            d: int8x16_t,
            n: i64,
            ptr: *const i8,
        ) -> int8x16x4_t;
    }
    _vld4q_lane_s8(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_s64<const LANE: i32>(a: *const i64, b: int64x2x4_t) -> int64x2x4_t {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.ld4lane.v2i64.p0i8"
        )]
        fn _vld4q_lane_s64(
            a: int64x2_t,
            b: int64x2_t,
            c: int64x2_t,
            d: int64x2_t,
            n: i64,
            ptr: *const i8,
        ) -> int64x2x4_t;
    }
    _vld4q_lane_s64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_p64<const LANE: i32>(a: *const p64, b: poly64x2x4_t) -> poly64x2x4_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld4q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_u8<const LANE: i32>(a: *const u8, b: uint8x16x4_t) -> uint8x16x4_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld4q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_u64<const LANE: i32>(a: *const u64, b: uint64x2x4_t) -> uint64x2x4_t {
    static_assert_uimm_bits!(LANE, 1);
    transmute(vld4q_lane_s64::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ld4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vld4q_lane_p8<const LANE: i32>(a: *const p8, b: poly8x16x4_t) -> poly8x16x4_t {
    static_assert_uimm_bits!(LANE, 4);
    transmute(vld4q_lane_s8::<LANE>(transmute(a), transmute(b)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(ld4))]
pub unsafe fn vld4q_p64(a: *const p64) -> poly64x2x4_t {
    transmute(vld4q_s64(transmute(a)))
}

#[doc = "Load multiple 4-element structures to four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vld4q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ld4))]
pub unsafe fn vld4q_u64(a: *const u64) -> uint64x2x4_t {
    transmute(vld4q_s64(transmute(a)))
}

#[doc = "Maximum (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmax_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmax))]
pub unsafe fn vmax_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmax.v1f64"
        )]
        fn _vmax_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vmax_f64(a, b)
}

#[doc = "Maximum (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmax))]
pub unsafe fn vmaxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmax.v2f64"
        )]
        fn _vmaxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vmaxq_f64(a, b)
}

#[doc = "Floating-point Maximum Number (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxnm_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxnm))]
pub unsafe fn vmaxnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnm.v1f64"
        )]
        fn _vmaxnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vmaxnm_f64(a, b)
}

#[doc = "Floating-point Maximum Number (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxnmq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxnm))]
pub unsafe fn vmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnm.v2f64"
        )]
        fn _vmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vmaxnmq_f64(a, b)
}

#[doc = "Floating-point maximum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxnmv_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
pub unsafe fn vmaxnmv_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmv.f32.v2f32"
        )]
        fn _vmaxnmv_f32(a: float32x2_t) -> f32;
    }
    _vmaxnmv_f32(a)
}

#[doc = "Floating-point maximum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxnmvq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
pub unsafe fn vmaxnmvq_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmv.f64.v2f64"
        )]
        fn _vmaxnmvq_f64(a: float64x2_t) -> f64;
    }
    _vmaxnmvq_f64(a)
}

#[doc = "Floating-point maximum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmaxnmvq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxnmv))]
pub unsafe fn vmaxnmvq_f32(a: float32x4_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmv.f32.v4f32"
        )]
        fn _vmaxnmvq_f32(a: float32x4_t) -> f32;
    }
    _vmaxnmvq_f32(a)
}

#[doc = "Minimum (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmin_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmin))]
pub unsafe fn vmin_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmin.v1f64"
        )]
        fn _vmin_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vmin_f64(a, b)
}

#[doc = "Minimum (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmin))]
pub unsafe fn vminq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmin.v2f64"
        )]
        fn _vminq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vminq_f64(a, b)
}

#[doc = "Floating-point Minimum Number (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminnm_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fminnm))]
pub unsafe fn vminnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnm.v1f64"
        )]
        fn _vminnm_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vminnm_f64(a, b)
}

#[doc = "Floating-point Minimum Number (vector)"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminnmq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fminnm))]
pub unsafe fn vminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnm.v2f64"
        )]
        fn _vminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vminnmq_f64(a, b)
}

#[doc = "Floating-point minimum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminnmv_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vminnmv_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmv.f32.v2f32"
        )]
        fn _vminnmv_f32(a: float32x2_t) -> f32;
    }
    _vminnmv_f32(a)
}

#[doc = "Floating-point minimum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminnmvq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vminnmvq_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmv.f64.v2f64"
        )]
        fn _vminnmvq_f64(a: float64x2_t) -> f64;
    }
    _vminnmvq_f64(a)
}

#[doc = "Floating-point minimum number across vector"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vminnmvq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmv))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vminnmvq_f32(a: float32x4_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmv.f32.v4f32"
        )]
        fn _vminnmvq_f32(a: float32x4_t) -> f32;
    }
    _vminnmvq_f32(a)
}

#[doc = "Floating-point multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmla_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmla_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    simd_add(a, simd_mul(b, c))
}

#[doc = "Floating-point multiply-add to accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlaq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlaq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    simd_add(a, simd_mul(b, c))
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_lane_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlal_high_s16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_laneq_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmlal_high_s16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_lane_s32<const LANE: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmlal_high_s32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_laneq_s32<const LANE: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlal_high_s32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_lane_u16<const LANE: i32>(
    a: uint32x4_t,
    b: uint16x8_t,
    c: uint16x4_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlal_high_u16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_laneq_u16<const LANE: i32>(
    a: uint32x4_t,
    b: uint16x8_t,
    c: uint16x8_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmlal_high_u16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_lane_u32<const LANE: i32>(
    a: uint64x2_t,
    b: uint32x4_t,
    c: uint32x2_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmlal_high_u32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_laneq_u32<const LANE: i32>(
    a: uint64x2_t,
    b: uint32x4_t,
    c: uint32x4_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlal_high_u32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vmlal_high_s16(a, b, vdupq_n_s16(c))
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vmlal_high_s32(a, b, vdupq_n_s32(c))
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_n_u16(a: uint32x4_t, b: uint16x8_t, c: u16) -> uint32x4_t {
    vmlal_high_u16(a, b, vdupq_n_u16(c))
}

#[doc = "Multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_n_u32(a: uint64x2_t, b: uint32x4_t, c: u32) -> uint64x2_t {
    vmlal_high_u32(a, b, vdupq_n_u32(c))
}

#[doc = "Signed multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let b: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: int8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlal_s8(a, b, c)
}

#[doc = "Signed multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let b: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let c: int16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    vmlal_s16(a, b, c)
}

#[doc = "Signed multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let b: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    let c: int32x2_t = simd_shuffle!(c, c, [2, 3]);
    vmlal_s32(a, b, c)
}

#[doc = "Unsigned multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let b: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: uint8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlal_u8(a, b, c)
}

#[doc = "Unsigned multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let b: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let c: uint16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    vmlal_u16(a, b, c)
}

#[doc = "Unsigned multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlal_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlal_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let b: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    let c: uint32x2_t = simd_shuffle!(c, c, [2, 3]);
    vmlal_u32(a, b, c)
}

#[doc = "Floating-point multiply-subtract from accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmls_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmls_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t) -> float64x1_t {
    simd_sub(a, simd_mul(b, c))
}

#[doc = "Floating-point multiply-subtract from accumulator"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsq_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t) -> float64x2_t {
    simd_sub(a, simd_mul(b, c))
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_lane_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlsl_high_s16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_laneq_s16<const LANE: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmlsl_high_s16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_lane_s32<const LANE: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmlsl_high_s32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_laneq_s32<const LANE: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlsl_high_s32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_lane_u16<const LANE: i32>(
    a: uint32x4_t,
    b: uint16x8_t,
    c: uint16x4_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlsl_high_u16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_laneq_u16<const LANE: i32>(
    a: uint32x4_t,
    b: uint16x8_t,
    c: uint16x8_t,
) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmlsl_high_u16(
        a,
        b,
        simd_shuffle!(
            c,
            c,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_lane_u32<const LANE: i32>(
    a: uint64x2_t,
    b: uint32x4_t,
    c: uint32x2_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmlsl_high_u32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_laneq_u32<const LANE: i32>(
    a: uint64x2_t,
    b: uint32x4_t,
    c: uint32x4_t,
) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmlsl_high_u32(
        a,
        b,
        simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vmlsl_high_s16(a, b, vdupq_n_s16(c))
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vmlsl_high_s32(a, b, vdupq_n_s32(c))
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_n_u16(a: uint32x4_t, b: uint16x8_t, c: u16) -> uint32x4_t {
    vmlsl_high_u16(a, b, vdupq_n_u16(c))
}

#[doc = "Multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_n_u32(a: uint64x2_t, b: uint32x4_t, c: u32) -> uint64x2_t {
    vmlsl_high_u32(a, b, vdupq_n_u32(c))
}

#[doc = "Signed multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_s8(a: int16x8_t, b: int8x16_t, c: int8x16_t) -> int16x8_t {
    let b: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: int8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlsl_s8(a, b, c)
}

#[doc = "Signed multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    let b: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let c: int16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    vmlsl_s16(a, b, c)
}

#[doc = "Signed multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    let b: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    let c: int32x2_t = simd_shuffle!(c, c, [2, 3]);
    vmlsl_s32(a, b, c)
}

#[doc = "Unsigned multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_u8(a: uint16x8_t, b: uint8x16_t, c: uint8x16_t) -> uint16x8_t {
    let b: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let c: uint8x8_t = simd_shuffle!(c, c, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmlsl_u8(a, b, c)
}

#[doc = "Unsigned multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_u16(a: uint32x4_t, b: uint16x8_t, c: uint16x8_t) -> uint32x4_t {
    let b: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let c: uint16x4_t = simd_shuffle!(c, c, [4, 5, 6, 7]);
    vmlsl_u16(a, b, c)
}

#[doc = "Unsigned multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmlsl_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmlsl_high_u32(a: uint64x2_t, b: uint32x4_t, c: uint32x4_t) -> uint64x2_t {
    let b: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    let c: uint32x2_t = simd_shuffle!(c, c, [2, 3]);
    vmlsl_u32(a, b, c)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sxtl2))]
pub unsafe fn vmovl_high_s8(a: int8x16_t) -> int16x8_t {
    let a: int8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmovl_s8(a)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sxtl2))]
pub unsafe fn vmovl_high_s16(a: int16x8_t) -> int32x4_t {
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    vmovl_s16(a)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sxtl2))]
pub unsafe fn vmovl_high_s32(a: int32x4_t) -> int64x2_t {
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    vmovl_s32(a)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uxtl2))]
pub unsafe fn vmovl_high_u8(a: uint8x16_t) -> uint16x8_t {
    let a: uint8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmovl_u8(a)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uxtl2))]
pub unsafe fn vmovl_high_u16(a: uint16x8_t) -> uint32x4_t {
    let a: uint16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    vmovl_u16(a)
}

#[doc = "Vector move"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovl_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uxtl2))]
pub unsafe fn vmovl_high_u32(a: uint32x4_t) -> uint64x2_t {
    let a: uint32x2_t = simd_shuffle!(a, a, [2, 3]);
    vmovl_u32(a)
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s16(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    let c: int8x8_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s32(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    let c: int16x4_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_s64(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    let c: int32x2_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3])
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u16(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    let c: uint8x8_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u32(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    let c: uint16x4_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmovn_high_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(xtn2))]
pub unsafe fn vmovn_high_u64(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    let c: uint32x2_t = simd_cast(b);
    simd_shuffle!(a, c, [0, 1, 2, 3])
}

#[doc = "Multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmul_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmul_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_mul(a, b)
}

#[doc = "Multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmul))]
pub unsafe fn vmulq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_mul(a, b)
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmul_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmul_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE == 0);
    simd_mul(a, transmute::<f64, _>(simd_extract!(b, LANE as u32)))
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmul_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmul_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x2_t) -> float64x1_t {
    static_assert_uimm_bits!(LANE, 1);
    simd_mul(a, transmute::<f64, _>(simd_extract!(b, LANE as u32)))
}

#[doc = "Vector multiply by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmul_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmul_n_f64(a: float64x1_t, b: f64) -> float64x1_t {
    simd_mul(a, vdup_n_f64(b))
}

#[doc = "Vector multiply by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulq_n_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulq_n_f64(a: float64x2_t, b: f64) -> float64x2_t {
    simd_mul(a, vdupq_n_f64(b))
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmuld_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmuld_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> f64 {
    static_assert!(LANE == 0);
    let b: f64 = simd_extract!(b, LANE as u32);
    a * b
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x4_t) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmull_high_s16(
        a,
        simd_shuffle!(
            b,
            b,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmull_high_s16(
        a,
        simd_shuffle!(
            b,
            b,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x2_t) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmull_high_s32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmull_high_s32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_lane_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_lane_u16<const LANE: i32>(a: uint16x8_t, b: uint16x4_t) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmull_high_u16(
        a,
        simd_shuffle!(
            b,
            b,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_laneq_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_laneq_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    static_assert_uimm_bits!(LANE, 3);
    vmull_high_u16(
        a,
        simd_shuffle!(
            b,
            b,
            [
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32,
                LANE as u32
            ]
        ),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_lane_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_lane_u32<const LANE: i32>(a: uint32x4_t, b: uint32x2_t) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmull_high_u32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_laneq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_laneq_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmull_high_u32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_n_s16(a: int16x8_t, b: i16) -> int32x4_t {
    vmull_high_s16(a, vdupq_n_s16(b))
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(smull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_n_s32(a: int32x4_t, b: i32) -> int64x2_t {
    vmull_high_s32(a, vdupq_n_s32(b))
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_n_u16(a: uint16x8_t, b: u16) -> uint32x4_t {
    vmull_high_u16(a, vdupq_n_u16(b))
}

#[doc = "Multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(umull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmull_high_n_u32(a: uint32x4_t, b: u32) -> uint64x2_t {
    vmull_high_u32(a, vdupq_n_u32(b))
}

#[doc = "Polynomial multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_high_p64(a: poly64x2_t, b: poly64x2_t) -> p128 {
    vmull_p64(simd_extract!(a, 1), simd_extract!(b, 1))
}

#[doc = "Polynomial multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_high_p8(a: poly8x16_t, b: poly8x16_t) -> poly16x8_t {
    let a: poly8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: poly8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_p8(a, b)
}

#[doc = "Signed multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let a: int8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_s8(a, b)
}

#[doc = "Signed multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    vmull_s16(a, b)
}

#[doc = "Signed multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(smull2))]
pub unsafe fn vmull_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    vmull_s32(a, b)
}

#[doc = "Unsigned multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let a: uint8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let b: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    vmull_u8(a, b)
}

#[doc = "Unsigned multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let a: uint16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    vmull_u16(a, b)
}

#[doc = "Unsigned multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(umull2))]
pub unsafe fn vmull_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let a: uint32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    vmull_u32(a, b)
}

#[doc = "Polynomial multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmull_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(pmull))]
pub unsafe fn vmull_p64(a: p64, b: p64) -> p128 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.pmull64"
        )]
        fn _vmull_p64(a: p64, b: p64) -> int8x16_t;
    }
    transmute(_vmull_p64(a, b))
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x1_t) -> float64x2_t {
    static_assert!(LANE == 0);
    simd_mul(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    simd_mul(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmuls_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmuls_lane_f32<const LANE: i32>(a: f32, b: float32x2_t) -> f32 {
    static_assert_uimm_bits!(LANE, 1);
    let b: f32 = simd_extract!(b, LANE as u32);
    a * b
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmuls_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmuls_laneq_f32<const LANE: i32>(a: f32, b: float32x4_t) -> f32 {
    static_assert_uimm_bits!(LANE, 2);
    let b: f32 = simd_extract!(b, LANE as u32);
    a * b
}

#[doc = "Floating-point multiply"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmuld_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmuld_laneq_f64<const LANE: i32>(a: f64, b: float64x2_t) -> f64 {
    static_assert_uimm_bits!(LANE, 1);
    let b: f64 = simd_extract!(b, LANE as u32);
    a * b
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulx_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.v2f32"
        )]
        fn _vmulx_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    _vmulx_f32(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.v4f32"
        )]
        fn _vmulxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vmulxq_f32(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulx_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.v1f64"
        )]
        fn _vmulx_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vmulx_f64(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.v2f64"
        )]
        fn _vmulxq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vmulxq_f64(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulx_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmulx_f32(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulx_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x4_t) -> float32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    vmulx_f32(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x2_t) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    vmulxq_f32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vmulxq_f32(
        a,
        simd_shuffle!(b, b, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]),
    )
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxq_laneq_f64<const LANE: i32>(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vmulxq_f64(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulx_lane_f64<const LANE: i32>(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE == 0);
    vmulx_f64(a, transmute::<f64, _>(simd_extract!(b, LANE as u32)))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulx_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulx_laneq_f64<const LANE: i32>(a: float64x1_t, b: float64x2_t) -> float64x1_t {
    static_assert_uimm_bits!(LANE, 1);
    vmulx_f64(a, transmute::<f64, _>(simd_extract!(b, LANE as u32)))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxd_f64(a: f64, b: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.f64"
        )]
        fn _vmulxd_f64(a: f64, b: f64) -> f64;
    }
    _vmulxd_f64(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmulx))]
pub unsafe fn vmulxs_f32(a: f32, b: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmulx.f32"
        )]
        fn _vmulxs_f32(a: f32, b: f32) -> f32;
    }
    _vmulxs_f32(a, b)
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxd_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxd_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> f64 {
    static_assert!(LANE == 0);
    vmulxd_f64(a, simd_extract!(b, LANE as u32))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxd_laneq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxd_laneq_f64<const LANE: i32>(a: f64, b: float64x2_t) -> f64 {
    static_assert_uimm_bits!(LANE, 1);
    vmulxd_f64(a, simd_extract!(b, LANE as u32))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxs_lane_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxs_lane_f32<const LANE: i32>(a: f32, b: float32x2_t) -> f32 {
    static_assert_uimm_bits!(LANE, 1);
    vmulxs_f32(a, simd_extract!(b, LANE as u32))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxs_laneq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxs_laneq_f32<const LANE: i32>(a: f32, b: float32x4_t) -> f32 {
    static_assert_uimm_bits!(LANE, 2);
    vmulxs_f32(a, simd_extract!(b, LANE as u32))
}

#[doc = "Floating-point multiply extended"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vmulxq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmulx, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vmulxq_lane_f64<const LANE: i32>(a: float64x2_t, b: float64x1_t) -> float64x2_t {
    static_assert!(LANE == 0);
    vmulxq_f64(a, simd_shuffle!(b, b, [LANE as u32, LANE as u32]))
}

#[doc = "Negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vneg_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fneg))]
pub unsafe fn vneg_f64(a: float64x1_t) -> float64x1_t {
    simd_neg(a)
}

#[doc = "Negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vnegq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fneg))]
pub unsafe fn vnegq_f64(a: float64x2_t) -> float64x2_t {
    simd_neg(a)
}

#[doc = "Negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vneg_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(neg))]
pub unsafe fn vneg_s64(a: int64x1_t) -> int64x1_t {
    simd_neg(a)
}

#[doc = "Negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vnegq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(neg))]
pub unsafe fn vnegq_s64(a: int64x2_t) -> int64x2_t {
    simd_neg(a)
}

#[doc = "Negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vnegd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(neg))]
pub unsafe fn vnegd_s64(a: i64) -> i64 {
    a.wrapping_neg()
}

#[doc = "Floating-point add pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpaddd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vpaddd_f64(a: float64x2_t) -> f64 {
    let a1: f64 = simd_extract!(a, 0);
    let a2: f64 = simd_extract!(a, 1);
    a1 + a2
}

#[doc = "Floating-point add pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpadds_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vpadds_f32(a: float32x2_t) -> f32 {
    let a1: f32 = simd_extract!(a, 0);
    let a2: f32 = simd_extract!(a, 1);
    a1 + a2
}

#[doc = "Floating-point add pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpaddq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(faddp))]
pub unsafe fn vpaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.faddp.v4f32"
        )]
        fn _vpaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vpaddq_f32(a, b)
}

#[doc = "Floating-point add pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpaddq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(faddp))]
pub unsafe fn vpaddq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.faddp.v2f64"
        )]
        fn _vpaddq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vpaddq_f64(a, b)
}

#[doc = "Floating-point Maximum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxnm_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpmaxnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmp.v2f32"
        )]
        fn _vpmaxnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    _vpmaxnm_f32(a, b)
}

#[doc = "Floating-point Maximum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxnmq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpmaxnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmp.v4f32"
        )]
        fn _vpmaxnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vpmaxnmq_f32(a, b)
}

#[doc = "Floating-point Maximum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxnmq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmp.v2f64"
        )]
        fn _vpmaxnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vpmaxnmq_f64(a, b)
}

#[doc = "Floating-point maximum number pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxnmqd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpmaxnmqd_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmv.f64.v2f64"
        )]
        fn _vpmaxnmqd_f64(a: float64x2_t) -> f64;
    }
    _vpmaxnmqd_f64(a)
}

#[doc = "Floating-point maximum number pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxnms_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fmaxnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpmaxnms_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxnmv.f32.v2f32"
        )]
        fn _vpmaxnms_f32(a: float32x2_t) -> f32;
    }
    _vpmaxnms_f32(a)
}

#[doc = "Floating-point maximum pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxqd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vpmaxqd_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxv.f64.v2f64"
        )]
        fn _vpmaxqd_f64(a: float64x2_t) -> f64;
    }
    _vpmaxqd_f64(a)
}

#[doc = "Floating-point maximum pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmaxs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fmaxp))]
pub unsafe fn vpmaxs_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fmaxv.f32.v2f32"
        )]
        fn _vpmaxs_f32(a: float32x2_t) -> f32;
    }
    _vpmaxs_f32(a)
}

#[doc = "Floating-point Minimum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminnm_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpminnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmp.v2f32"
        )]
        fn _vpminnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
    _vpminnm_f32(a, b)
}

#[doc = "Floating-point Minimum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminnmq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpminnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmp.v4f32"
        )]
        fn _vpminnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
    _vpminnmq_f32(a, b)
}

#[doc = "Floating-point Minimum Number Pairwise (vector)."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminnmq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmp.v2f64"
        )]
        fn _vpminnmq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vpminnmq_f64(a, b)
}

#[doc = "Floating-point minimum number pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminnmqd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpminnmqd_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmv.f64.v2f64"
        )]
        fn _vpminnmqd_f64(a: float64x2_t) -> f64;
    }
    _vpminnmqd_f64(a)
}

#[doc = "Floating-point minimum number pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminnms_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fminnmp))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vpminnms_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminnmv.f32.v2f32"
        )]
        fn _vpminnms_f32(a: float32x2_t) -> f32;
    }
    _vpminnms_f32(a)
}

#[doc = "Floating-point minimum pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpminqd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vpminqd_f64(a: float64x2_t) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminv.f64.v2f64"
        )]
        fn _vpminqd_f64(a: float64x2_t) -> f64;
    }
    _vpminqd_f64(a)
}

#[doc = "Floating-point minimum pairwise"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vpmins_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fminp))]
pub unsafe fn vpmins_f32(a: float32x2_t) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.fminv.f32.v2f32"
        )]
        fn _vpmins_f32(a: float32x2_t) -> f32;
    }
    _vpmins_f32(a)
}

#[doc = "Signed saturating Absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabs_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabs_s64(a: int64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqabs.v1i64"
        )]
        fn _vqabs_s64(a: int64x1_t) -> int64x1_t;
    }
    _vqabs_s64(a)
}

#[doc = "Signed saturating Absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabsq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabsq_s64(a: int64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqabs.v2i64"
        )]
        fn _vqabsq_s64(a: int64x2_t) -> int64x2_t;
    }
    _vqabsq_s64(a)
}

#[doc = "Signed saturating absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabsb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabsb_s8(a: i8) -> i8 {
    simd_extract!(vqabs_s8(vdup_n_s8(a)), 0)
}

#[doc = "Signed saturating absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabsh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabsh_s16(a: i16) -> i16 {
    simd_extract!(vqabs_s16(vdup_n_s16(a)), 0)
}

#[doc = "Signed saturating absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabss_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabss_s32(a: i32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqabs.i32"
        )]
        fn _vqabss_s32(a: i32) -> i32;
    }
    _vqabss_s32(a)
}

#[doc = "Signed saturating absolute value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqabsd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sqabs))]
pub unsafe fn vqabsd_s64(a: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqabs.i64"
        )]
        fn _vqabsd_s64(a: i64) -> i64;
    }
    _vqabsd_s64(a)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract!(vqadd_s8(a, b), 0)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqadd_s16(a, b), 0)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddb_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddb_u8(a: u8, b: u8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: uint8x8_t = vdup_n_u8(b);
    simd_extract!(vqadd_u8(a, b), 0)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddh_u16(a: u16, b: u16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: uint16x4_t = vdup_n_u16(b);
    simd_extract!(vqadd_u16(a, b), 0)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqadds_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqadds_s32(a: i32, b: i32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqadd.i32"
        )]
        fn _vqadds_s32(a: i32, b: i32) -> i32;
    }
    _vqadds_s32(a, b)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqadd))]
pub unsafe fn vqaddd_s64(a: i64, b: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqadd.i64"
        )]
        fn _vqaddd_s64(a: i64, b: i64) -> i64;
    }
    _vqaddd_s64(a, b)
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqadds_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqadds_u32(a: u32, b: u32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqadd.i32"
        )]
        fn _vqadds_u32(a: i32, b: i32) -> i32;
    }
    _vqadds_u32(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Saturating add"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqaddd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqadd))]
pub unsafe fn vqaddd_u64(a: u64, b: u64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqadd.i64"
        )]
        fn _vqaddd_u64(a: i64, b: i64) -> i64;
    }
    _vqaddd_u64(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_lane_s16<const N: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 2);
    vqaddq_s32(a, vqdmull_high_lane_s16::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_laneq_s16<const N: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    vqaddq_s32(a, vqdmull_high_laneq_s16::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_lane_s32<const N: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 1);
    vqaddq_s64(a, vqdmull_high_lane_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_laneq_s32<const N: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    vqaddq_s64(a, vqdmull_high_laneq_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vqaddq_s32(a, vqdmull_high_n_s16(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    vqaddq_s32(a, vqdmull_high_s16(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vqaddq_s64(a, vqdmull_high_n_s32(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    vqaddq_s64(a, vqdmull_high_s32(b, c))
}

#[doc = "Vector widening saturating doubling multiply accumulate with scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, N = 2))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_laneq_s16<const N: i32>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    vqaddq_s32(a, vqdmull_laneq_s16::<N>(b, c))
}

#[doc = "Vector widening saturating doubling multiply accumulate with scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlal_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlal_laneq_s32<const N: i32>(
    a: int64x2_t,
    b: int32x2_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    vqaddq_s64(a, vqdmull_laneq_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlalh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlalh_lane_s16<const LANE: i32>(a: i32, b: i16, c: int16x4_t) -> i32 {
    static_assert_uimm_bits!(LANE, 2);
    vqdmlalh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlalh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlalh_laneq_s16<const LANE: i32>(a: i32, b: i16, c: int16x8_t) -> i32 {
    static_assert_uimm_bits!(LANE, 3);
    vqdmlalh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlals_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlals_lane_s32<const LANE: i32>(a: i64, b: i32, c: int32x2_t) -> i64 {
    static_assert_uimm_bits!(LANE, 1);
    vqdmlals_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlals_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlals_laneq_s32<const LANE: i32>(a: i64, b: i32, c: int32x4_t) -> i64 {
    static_assert_uimm_bits!(LANE, 2);
    vqdmlals_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlalh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlalh_s16(a: i32, b: i16, c: i16) -> i32 {
    let x: int32x4_t = vqdmull_s16(vdup_n_s16(b), vdup_n_s16(c));
    vqadds_s32(a, simd_extract!(x, 0))
}

#[doc = "Signed saturating doubling multiply-add long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlals_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlal))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlals_s32(a: i64, b: i32, c: i32) -> i64 {
    let x: i64 = vqaddd_s64(a, vqdmulls_s32(b, c));
    x as i64
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_lane_s16<const N: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 2);
    vqsubq_s32(a, vqdmull_high_lane_s16::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_laneq_s16<const N: i32>(
    a: int32x4_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    vqsubq_s32(a, vqdmull_high_laneq_s16::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_lane_s32<const N: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 1);
    vqsubq_s64(a, vqdmull_high_lane_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_laneq_s32<const N: i32>(
    a: int64x2_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    vqsubq_s64(a, vqdmull_high_laneq_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_n_s16(a: int32x4_t, b: int16x8_t, c: i16) -> int32x4_t {
    vqsubq_s32(a, vqdmull_high_n_s16(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_s16(a: int32x4_t, b: int16x8_t, c: int16x8_t) -> int32x4_t {
    vqsubq_s32(a, vqdmull_high_s16(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_n_s32(a: int64x2_t, b: int32x4_t, c: i32) -> int64x2_t {
    vqsubq_s64(a, vqdmull_high_n_s32(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_high_s32(a: int64x2_t, b: int32x4_t, c: int32x4_t) -> int64x2_t {
    vqsubq_s64(a, vqdmull_high_s32(b, c))
}

#[doc = "Vector widening saturating doubling multiply subtract with scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, N = 2))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_laneq_s16<const N: i32>(
    a: int32x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    vqsubq_s32(a, vqdmull_laneq_s16::<N>(b, c))
}

#[doc = "Vector widening saturating doubling multiply subtract with scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsl_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, N = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsl_laneq_s32<const N: i32>(
    a: int64x2_t,
    b: int32x2_t,
    c: int32x4_t,
) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    vqsubq_s64(a, vqdmull_laneq_s32::<N>(b, c))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlslh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlslh_lane_s16<const LANE: i32>(a: i32, b: i16, c: int16x4_t) -> i32 {
    static_assert_uimm_bits!(LANE, 2);
    vqdmlslh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlslh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlslh_laneq_s16<const LANE: i32>(a: i32, b: i16, c: int16x8_t) -> i32 {
    static_assert_uimm_bits!(LANE, 3);
    vqdmlslh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsls_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsls_lane_s32<const LANE: i32>(a: i64, b: i32, c: int32x2_t) -> i64 {
    static_assert_uimm_bits!(LANE, 1);
    vqdmlsls_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsls_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl, LANE = 0))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsls_laneq_s32<const LANE: i32>(a: i64, b: i32, c: int32x4_t) -> i64 {
    static_assert_uimm_bits!(LANE, 2);
    vqdmlsls_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlslh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlslh_s16(a: i32, b: i16, c: i16) -> i32 {
    let x: int32x4_t = vqdmull_s16(vdup_n_s16(b), vdup_n_s16(c));
    vqsubs_s32(a, simd_extract!(x, 0))
}

#[doc = "Signed saturating doubling multiply-subtract long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmlsls_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmlsl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmlsls_s32(a: i64, b: i32, c: i32) -> i64 {
    let x: i64 = vqsubd_s64(a, vqdmulls_s32(b, c));
    x as i64
}

#[doc = "Vector saturating doubling multiply high by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulh_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_uimm_bits!(LANE, 2);
    vqdmulh_s16(a, vdup_n_s16(simd_extract!(b, LANE as u32)))
}

#[doc = "Vector saturating doubling multiply high by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhq_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x4_t) -> int16x8_t {
    static_assert_uimm_bits!(LANE, 2);
    vqdmulhq_s16(a, vdupq_n_s16(simd_extract!(b, LANE as u32)))
}

#[doc = "Vector saturating doubling multiply high by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulh_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulh_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    vqdmulh_s32(a, vdup_n_s32(simd_extract!(b, LANE as u32)))
}

#[doc = "Vector saturating doubling multiply high by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhq_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x2_t) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    vqdmulhq_s32(a, vdupq_n_s32(simd_extract!(b, LANE as u32)))
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhh_lane_s16<const N: i32>(a: i16, b: int16x4_t) -> i16 {
    static_assert_uimm_bits!(N, 2);
    let b: i16 = simd_extract!(b, N as u32);
    vqdmulhh_s16(a, b)
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhh_laneq_s16<const N: i32>(a: i16, b: int16x8_t) -> i16 {
    static_assert_uimm_bits!(N, 3);
    let b: i16 = simd_extract!(b, N as u32);
    vqdmulhh_s16(a, b)
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqdmulh_s16(a, b), 0)
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhs_s32(a: i32, b: i32) -> i32 {
    let a: int32x2_t = vdup_n_s32(a);
    let b: int32x2_t = vdup_n_s32(b);
    simd_extract!(vqdmulh_s32(a, b), 0)
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhs_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhs_lane_s32<const N: i32>(a: i32, b: int32x2_t) -> i32 {
    static_assert_uimm_bits!(N, 1);
    let b: i32 = simd_extract!(b, N as u32);
    vqdmulhs_s32(a, b)
}

#[doc = "Signed saturating doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulhs_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmulh, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulhs_laneq_s32<const N: i32>(a: i32, b: int32x4_t) -> i32 {
    static_assert_uimm_bits!(N, 2);
    let b: i32 = simd_extract!(b, N as u32);
    vqdmulhs_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_lane_s16<const N: i32>(a: int16x8_t, b: int16x4_t) -> int32x4_t {
    static_assert_uimm_bits!(N, 2);
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle!(b, b, [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_laneq_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle!(b, b, [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_lane_s32<const N: i32>(a: int32x4_t, b: int32x2_t) -> int64x2_t {
    static_assert_uimm_bits!(N, 1);
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle!(b, b, [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2, N = 4))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_laneq_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle!(b, b, [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_n_s16(a: int16x8_t, b: i16) -> int32x4_t {
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = vdup_n_s16(b);
    vqdmull_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_n_s32(a: int32x4_t, b: i32) -> int64x2_t {
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: int32x2_t = vdup_n_s32(b);
    vqdmull_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let a: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let b: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    vqdmull_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let a: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let b: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    vqdmull_s32(a, b)
}

#[doc = "Vector saturating doubling long multiply by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 4))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_laneq_s16<const N: i32>(a: int16x4_t, b: int16x8_t) -> int32x4_t {
    static_assert_uimm_bits!(N, 3);
    let b: int16x4_t = simd_shuffle!(b, b, [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

#[doc = "Vector saturating doubling long multiply by scalar"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmull_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmull_laneq_s32<const N: i32>(a: int32x2_t, b: int32x4_t) -> int64x2_t {
    static_assert_uimm_bits!(N, 2);
    let b: int32x2_t = simd_shuffle!(b, b, [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmullh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmullh_lane_s16<const N: i32>(a: i16, b: int16x4_t) -> i32 {
    static_assert_uimm_bits!(N, 2);
    let b: i16 = simd_extract!(b, N as u32);
    vqdmullh_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulls_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulls_laneq_s32<const N: i32>(a: i32, b: int32x4_t) -> i64 {
    static_assert_uimm_bits!(N, 2);
    let b: i32 = simd_extract!(b, N as u32);
    vqdmulls_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmullh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 4))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmullh_laneq_s16<const N: i32>(a: i16, b: int16x8_t) -> i32 {
    static_assert_uimm_bits!(N, 3);
    let b: i16 = simd_extract!(b, N as u32);
    vqdmullh_s16(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmullh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmullh_s16(a: i16, b: i16) -> i32 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqdmull_s16(a, b), 0)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulls_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull, N = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulls_lane_s32<const N: i32>(a: i32, b: int32x2_t) -> i64 {
    static_assert_uimm_bits!(N, 1);
    let b: i32 = simd_extract!(b, N as u32);
    vqdmulls_s32(a, b)
}

#[doc = "Signed saturating doubling multiply long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqdmulls_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqdmull))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqdmulls_s32(a: i32, b: i32) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqdmulls.scalar"
        )]
        fn _vqdmulls_s32(a: i32, b: i32) -> i64;
    }
    _vqdmulls_s32(a, b)
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_s16(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    simd_shuffle!(
        a,
        vqmovn_s16(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_s32(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    simd_shuffle!(a, vqmovn_s32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_s64(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    simd_shuffle!(a, vqmovn_s64(b), [0, 1, 2, 3])
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_u16(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        vqmovn_u16(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_u32(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    simd_shuffle!(a, vqmovn_u32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovn_high_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovn_high_u64(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    simd_shuffle!(a, vqmovn_u64(b), [0, 1, 2, 3])
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovnd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovnd_s64(a: i64) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.scalar.sqxtn.i32.i64"
        )]
        fn _vqmovnd_s64(a: i64) -> i32;
    }
    _vqmovnd_s64(a)
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovnd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovnd_u64(a: u64) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.scalar.uqxtn.i32.i64"
        )]
        fn _vqmovnd_u64(a: i64) -> i32;
    }
    _vqmovnd_u64(a.as_signed()).as_unsigned()
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovnh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovnh_s16(a: i16) -> i8 {
    simd_extract!(vqmovn_s16(vdupq_n_s16(a)), 0)
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovns_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovns_s32(a: i32) -> i16 {
    simd_extract!(vqmovn_s32(vdupq_n_s32(a)), 0)
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovnh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovnh_u16(a: u16) -> u8 {
    simd_extract!(vqmovn_u16(vdupq_n_u16(a)), 0)
}

#[doc = "Saturating extract narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovns_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqxtn))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovns_u32(a: u32) -> u16 {
    simd_extract!(vqmovn_u32(vdupq_n_u32(a)), 0)
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovun_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovun_high_s16(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        vqmovun_s16(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovun_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovun_high_s32(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    simd_shuffle!(a, vqmovun_s32(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovun_high_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovun_high_s64(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    simd_shuffle!(a, vqmovun_s64(b), [0, 1, 2, 3])
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovunh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovunh_s16(a: i16) -> u8 {
    simd_extract!(vqmovun_s16(vdupq_n_s16(a)), 0)
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovuns_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovuns_s32(a: i32) -> u16 {
    simd_extract!(vqmovun_s32(vdupq_n_s32(a)), 0)
}

#[doc = "Signed saturating extract unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqmovund_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqxtun))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqmovund_s64(a: i64) -> u32 {
    simd_extract!(vqmovun_s64(vdupq_n_s64(a)), 0)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqneg_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqneg_s64(a: int64x1_t) -> int64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqneg.v1i64"
        )]
        fn _vqneg_s64(a: int64x1_t) -> int64x1_t;
    }
    _vqneg_s64(a)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqnegq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegq_s64(a: int64x2_t) -> int64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqneg.v2i64"
        )]
        fn _vqnegq_s64(a: int64x2_t) -> int64x2_t;
    }
    _vqnegq_s64(a)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqnegb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegb_s8(a: i8) -> i8 {
    simd_extract!(vqneg_s8(vdup_n_s8(a)), 0)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqnegh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegh_s16(a: i16) -> i16 {
    simd_extract!(vqneg_s16(vdup_n_s16(a)), 0)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqnegs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegs_s32(a: i32) -> i32 {
    simd_extract!(vqneg_s32(vdup_n_s32(a)), 0)
}

#[doc = "Signed saturating negate"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqnegd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqneg))]
pub unsafe fn vqnegd_s64(a: i64) -> i64 {
    simd_extract!(vqneg_s64(vdup_n_s64(a)), 0)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_lane_s16<const LANE: i32>(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int16x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlah_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_lane_s32<const LANE: i32>(
    a: int32x2_t,
    b: int32x2_t,
    c: int32x2_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vqrdmlah_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_laneq_s16<const LANE: i32>(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE, 3);
    let c: int16x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlah_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_laneq_s32<const LANE: i32>(
    a: int32x2_t,
    b: int32x2_t,
    c: int32x4_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vqrdmlah_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_lane_s16<const LANE: i32>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int16x8_t = simd_shuffle!(
        c,
        c,
        [
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32
        ]
    );
    vqrdmlahq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_lane_s32<const LANE: i32>(
    a: int32x4_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlahq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_laneq_s16<const LANE: i32>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE, 3);
    let c: int16x8_t = simd_shuffle!(
        c,
        c,
        [
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32
        ]
    );
    vqrdmlahq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_laneq_s32<const LANE: i32>(
    a: int32x4_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlahq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlah.v4i16"
        )]
        fn _vqrdmlah_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t;
    }
    _vqrdmlah_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlah.v8i16"
        )]
        fn _vqrdmlahq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _vqrdmlahq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlah_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlah_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlah.v2i32"
        )]
        fn _vqrdmlah_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t;
    }
    _vqrdmlah_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlah.v4i32"
        )]
        fn _vqrdmlahq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vqrdmlahq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahh_lane_s16<const LANE: i32>(a: i16, b: i16, c: int16x4_t) -> i16 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmlahh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahh_laneq_s16<const LANE: i32>(a: i16, b: i16, c: int16x8_t) -> i16 {
    static_assert_uimm_bits!(LANE, 3);
    vqrdmlahh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahs_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahs_lane_s32<const LANE: i32>(a: i32, b: i32, c: int32x2_t) -> i32 {
    static_assert_uimm_bits!(LANE, 1);
    vqrdmlahs_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahs_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahs_laneq_s32<const LANE: i32>(a: i32, b: i32, c: int32x4_t) -> i32 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmlahs_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahh_s16(a: i16, b: i16, c: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    let c: int16x4_t = vdup_n_s16(c);
    simd_extract!(vqrdmlah_s16(a, b, c), 0)
}

#[doc = "Signed saturating rounding doubling multiply accumulate returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlahs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlah))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlahs_s32(a: i32, b: i32, c: i32) -> i32 {
    let a: int32x2_t = vdup_n_s32(a);
    let b: int32x2_t = vdup_n_s32(b);
    let c: int32x2_t = vdup_n_s32(c);
    simd_extract!(vqrdmlah_s32(a, b, c), 0)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_lane_s16<const LANE: i32>(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x4_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int16x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlsh_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_lane_s32<const LANE: i32>(
    a: int32x2_t,
    b: int32x2_t,
    c: int32x2_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vqrdmlsh_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_laneq_s16<const LANE: i32>(
    a: int16x4_t,
    b: int16x4_t,
    c: int16x8_t,
) -> int16x4_t {
    static_assert_uimm_bits!(LANE, 3);
    let c: int16x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlsh_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_laneq_s32<const LANE: i32>(
    a: int32x2_t,
    b: int32x2_t,
    c: int32x4_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vqrdmlsh_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_lane_s16<const LANE: i32>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x4_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int16x8_t = simd_shuffle!(
        c,
        c,
        [
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32
        ]
    );
    vqrdmlshq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_lane_s32<const LANE: i32>(
    a: int32x4_t,
    b: int32x4_t,
    c: int32x2_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 1);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlshq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_laneq_s16<const LANE: i32>(
    a: int16x8_t,
    b: int16x8_t,
    c: int16x8_t,
) -> int16x8_t {
    static_assert_uimm_bits!(LANE, 3);
    let c: int16x8_t = simd_shuffle!(
        c,
        c,
        [
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32,
            LANE as u32
        ]
    );
    vqrdmlshq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_laneq_s32<const LANE: i32>(
    a: int32x4_t,
    b: int32x4_t,
    c: int32x4_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmlshq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlsh.v4i16"
        )]
        fn _vqrdmlsh_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t;
    }
    _vqrdmlsh_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlsh.v8i16"
        )]
        fn _vqrdmlshq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t;
    }
    _vqrdmlshq_s16(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlsh_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlsh_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlsh.v2i32"
        )]
        fn _vqrdmlsh_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t;
    }
    _vqrdmlsh_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrdmlsh.v4i32"
        )]
        fn _vqrdmlshq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vqrdmlshq_s32(a, b, c)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshh_lane_s16<const LANE: i32>(a: i16, b: i16, c: int16x4_t) -> i16 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmlshh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshh_laneq_s16<const LANE: i32>(a: i16, b: i16, c: int16x8_t) -> i16 {
    static_assert_uimm_bits!(LANE, 3);
    vqrdmlshh_s16(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshs_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshs_lane_s32<const LANE: i32>(a: i32, b: i32, c: int32x2_t) -> i32 {
    static_assert_uimm_bits!(LANE, 1);
    vqrdmlshs_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshs_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshs_laneq_s32<const LANE: i32>(a: i32, b: i32, c: int32x4_t) -> i32 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmlshs_s32(a, b, simd_extract!(c, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshh_s16(a: i16, b: i16, c: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    let c: int16x4_t = vdup_n_s16(c);
    simd_extract!(vqrdmlsh_s16(a, b, c), 0)
}

#[doc = "Signed saturating rounding doubling multiply subtract returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmlshs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "rdm")]
#[cfg_attr(test, assert_instr(sqrdmlsh))]
#[stable(feature = "rdm_intrinsics", since = "1.62.0")]
pub unsafe fn vqrdmlshs_s32(a: i32, b: i32, c: i32) -> i32 {
    let a: int32x2_t = vdup_n_s32(a);
    let b: int32x2_t = vdup_n_s32(b);
    let c: int32x2_t = vdup_n_s32(c);
    simd_extract!(vqrdmlsh_s32(a, b, c), 0)
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhh_lane_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhh_lane_s16<const LANE: i32>(a: i16, b: int16x4_t) -> i16 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmulhh_s16(a, simd_extract!(b, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhh_laneq_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhh_laneq_s16<const LANE: i32>(a: i16, b: int16x8_t) -> i16 {
    static_assert_uimm_bits!(LANE, 3);
    vqrdmulhh_s16(a, simd_extract!(b, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhs_lane_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhs_lane_s32<const LANE: i32>(a: i32, b: int32x2_t) -> i32 {
    static_assert_uimm_bits!(LANE, 1);
    vqrdmulhs_s32(a, simd_extract!(b, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhs_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhs_laneq_s32<const LANE: i32>(a: i32, b: int32x4_t) -> i32 {
    static_assert_uimm_bits!(LANE, 2);
    vqrdmulhs_s32(a, simd_extract!(b, LANE as u32))
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhh_s16(a: i16, b: i16) -> i16 {
    simd_extract!(vqrdmulh_s16(vdup_n_s16(a), vdup_n_s16(b)), 0)
}

#[doc = "Signed saturating rounding doubling multiply returning high half"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrdmulhs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrdmulh))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrdmulhs_s32(a: i32, b: i32) -> i32 {
    simd_extract!(vqrdmulh_s32(vdup_n_s32(a), vdup_n_s32(b)), 0)
}

#[doc = "Signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshlb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshlb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract!(vqrshl_s8(a, b), 0)
}

#[doc = "Signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshlh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshlh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqrshl_s16(a, b), 0)
}

#[doc = "Unsigned signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshlb_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshlb_u8(a: u8, b: i8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract!(vqrshl_u8(a, b), 0)
}

#[doc = "Unsigned signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshlh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshlh_u16(a: u16, b: i16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqrshl_u16(a, b), 0)
}

#[doc = "Signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshld_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshld_s64(a: i64, b: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrshl.i64"
        )]
        fn _vqrshld_s64(a: i64, b: i64) -> i64;
    }
    _vqrshld_s64(a, b)
}

#[doc = "Signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshls_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshls_s32(a: i32, b: i32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqrshl.i32"
        )]
        fn _vqrshls_s32(a: i32, b: i32) -> i32;
    }
    _vqrshls_s32(a, b)
}

#[doc = "Unsigned signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshls_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshls_u32(a: u32, b: i32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqrshl.i32"
        )]
        fn _vqrshls_u32(a: i32, b: i32) -> i32;
    }
    _vqrshls_u32(a.as_signed(), b).as_unsigned()
}

#[doc = "Unsigned signed saturating rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshld_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshld_u64(a: u64, b: i64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqrshl.i64"
        )]
        fn _vqrshld_u64(a: i64, b: i64) -> i64;
    }
    _vqrshld_u64(a.as_signed(), b).as_unsigned()
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqrshrn_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqrshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqrshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqrshrn_n_u16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqrshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrn_high_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqrshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrnd_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrnd_n_u64<const N: i32>(a: u64) -> u32 {
    static_assert!(N >= 1 && N <= 32);
    let a: uint64x2_t = vdupq_n_u64(a);
    simd_extract!(vqrshrn_n_u64::<N>(a), 0)
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrnh_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrnh_n_u16<const N: i32>(a: u16) -> u8 {
    static_assert!(N >= 1 && N <= 8);
    let a: uint16x8_t = vdupq_n_u16(a);
    simd_extract!(vqrshrn_n_u16::<N>(a), 0)
}

#[doc = "Unsigned saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrns_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrns_n_u32<const N: i32>(a: u32) -> u16 {
    static_assert!(N >= 1 && N <= 16);
    let a: uint32x4_t = vdupq_n_u32(a);
    simd_extract!(vqrshrn_n_u32::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrnh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrnh_n_s16<const N: i32>(a: i16) -> i8 {
    static_assert!(N >= 1 && N <= 8);
    let a: int16x8_t = vdupq_n_s16(a);
    simd_extract!(vqrshrn_n_s16::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrns_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrns_n_s32<const N: i32>(a: i32) -> i16 {
    static_assert!(N >= 1 && N <= 16);
    let a: int32x4_t = vdupq_n_s32(a);
    simd_extract!(vqrshrn_n_s32::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrnd_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrnd_n_s64<const N: i32>(a: i64) -> i32 {
    static_assert!(N >= 1 && N <= 32);
    let a: int64x2_t = vdupq_n_s64(a);
    simd_extract!(vqrshrn_n_s64::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrun_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrun_high_n_s16<const N: i32>(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqrshrun_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrun_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrun_high_n_s32<const N: i32>(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqrshrun_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrun_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrun_high_n_s64<const N: i32>(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqrshrun_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrund_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrund_n_s64<const N: i32>(a: i64) -> u32 {
    static_assert!(N >= 1 && N <= 32);
    let a: int64x2_t = vdupq_n_s64(a);
    simd_extract!(vqrshrun_n_s64::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshrunh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshrunh_n_s16<const N: i32>(a: i16) -> u8 {
    static_assert!(N >= 1 && N <= 8);
    let a: int16x8_t = vdupq_n_s16(a);
    simd_extract!(vqrshrun_n_s16::<N>(a), 0)
}

#[doc = "Signed saturating rounded shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqrshruns_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqrshruns_n_s32<const N: i32>(a: i32) -> u16 {
    static_assert!(N >= 1 && N <= 16);
    let a: int32x4_t = vdupq_n_s32(a);
    simd_extract!(vqrshrun_n_s32::<N>(a), 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlb_n_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlb_n_s8<const N: i32>(a: i8) -> i8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(vqshl_n_s8::<N>(vdup_n_s8(a)), 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshld_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshld_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert_uimm_bits!(N, 6);
    simd_extract!(vqshl_n_s64::<N>(vdup_n_s64(a)), 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlh_n_s16<const N: i32>(a: i16) -> i16 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(vqshl_n_s16::<N>(vdup_n_s16(a)), 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshls_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshls_n_s32<const N: i32>(a: i32) -> i32 {
    static_assert_uimm_bits!(N, 5);
    simd_extract!(vqshl_n_s32::<N>(vdup_n_s32(a)), 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlb_n_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlb_n_u8<const N: i32>(a: u8) -> u8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(vqshl_n_u8::<N>(vdup_n_u8(a)), 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshld_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshld_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert_uimm_bits!(N, 6);
    simd_extract!(vqshl_n_u64::<N>(vdup_n_u64(a)), 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlh_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlh_n_u16<const N: i32>(a: u16) -> u16 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(vqshl_n_u16::<N>(vdup_n_u16(a)), 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshls_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshls_n_u32<const N: i32>(a: u32) -> u32 {
    static_assert_uimm_bits!(N, 5);
    simd_extract!(vqshl_n_u32::<N>(vdup_n_u32(a)), 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlb_s8(a: i8, b: i8) -> i8 {
    let c: int8x8_t = vqshl_s8(vdup_n_s8(a), vdup_n_s8(b));
    simd_extract!(c, 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlh_s16(a: i16, b: i16) -> i16 {
    let c: int16x4_t = vqshl_s16(vdup_n_s16(a), vdup_n_s16(b));
    simd_extract!(c, 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshls_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshls_s32(a: i32, b: i32) -> i32 {
    let c: int32x2_t = vqshl_s32(vdup_n_s32(a), vdup_n_s32(b));
    simd_extract!(c, 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlb_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlb_u8(a: u8, b: i8) -> u8 {
    let c: uint8x8_t = vqshl_u8(vdup_n_u8(a), vdup_n_s8(b));
    simd_extract!(c, 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlh_u16(a: u16, b: i16) -> u16 {
    let c: uint16x4_t = vqshl_u16(vdup_n_u16(a), vdup_n_s16(b));
    simd_extract!(c, 0)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshls_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshls_u32(a: u32, b: i32) -> u32 {
    let c: uint32x2_t = vqshl_u32(vdup_n_u32(a), vdup_n_s32(b));
    simd_extract!(c, 0)
}

#[doc = "Signed saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshld_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshld_s64(a: i64, b: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqshl.i64"
        )]
        fn _vqshld_s64(a: i64, b: i64) -> i64;
    }
    _vqshld_s64(a, b)
}

#[doc = "Unsigned saturating shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshld_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshld_u64(a: u64, b: i64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqshl.i64"
        )]
        fn _vqshld_u64(a: i64, b: i64) -> i64;
    }
    _vqshld_u64(a.as_signed(), b).as_unsigned()
}

#[doc = "Signed saturating shift left unsigned"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlub_n_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshlu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlub_n_s8<const N: i32>(a: i8) -> u8 {
    static_assert_uimm_bits!(N, 3);
    simd_extract!(vqshlu_n_s8::<N>(vdup_n_s8(a)), 0)
}

#[doc = "Signed saturating shift left unsigned"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlud_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshlu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlud_n_s64<const N: i32>(a: i64) -> u64 {
    static_assert_uimm_bits!(N, 6);
    simd_extract!(vqshlu_n_s64::<N>(vdup_n_s64(a)), 0)
}

#[doc = "Signed saturating shift left unsigned"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshluh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshlu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshluh_n_s16<const N: i32>(a: i16) -> u16 {
    static_assert_uimm_bits!(N, 4);
    simd_extract!(vqshlu_n_s16::<N>(vdup_n_s16(a)), 0)
}

#[doc = "Signed saturating shift left unsigned"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshlus_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshlu, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshlus_n_s32<const N: i32>(a: i32) -> u32 {
    static_assert_uimm_bits!(N, 5);
    simd_extract!(vqshlu_n_s32::<N>(vdup_n_s32(a)), 0)
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqshrn_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqshrn_n_u16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrn_high_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrnd_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrnd_n_s64<const N: i32>(a: i64) -> i32 {
    static_assert!(N >= 1 && N <= 32);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqshrn.i32"
        )]
        fn _vqshrnd_n_s64(a: i64, n: i32) -> i32;
    }
    _vqshrnd_n_s64(a, N)
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrnd_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrnd_n_u64<const N: i32>(a: u64) -> u32 {
    static_assert!(N >= 1 && N <= 32);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqshrn.i32"
        )]
        fn _vqshrnd_n_u64(a: i64, n: i32) -> i32;
    }
    _vqshrnd_n_u64(a.as_signed(), N).as_unsigned()
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrnh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrnh_n_s16<const N: i32>(a: i16) -> i8 {
    static_assert!(N >= 1 && N <= 8);
    simd_extract!(vqshrn_n_s16::<N>(vdupq_n_s16(a)), 0)
}

#[doc = "Signed saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrns_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrns_n_s32<const N: i32>(a: i32) -> i16 {
    static_assert!(N >= 1 && N <= 16);
    simd_extract!(vqshrn_n_s32::<N>(vdupq_n_s32(a)), 0)
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrnh_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrnh_n_u16<const N: i32>(a: u16) -> u8 {
    static_assert!(N >= 1 && N <= 8);
    simd_extract!(vqshrn_n_u16::<N>(vdupq_n_u16(a)), 0)
}

#[doc = "Unsigned saturating shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrns_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrns_n_u32<const N: i32>(a: u32) -> u16 {
    static_assert!(N >= 1 && N <= 16);
    simd_extract!(vqshrn_n_u32::<N>(vdupq_n_u32(a)), 0)
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrun_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrun_high_n_s16<const N: i32>(a: uint8x8_t, b: int16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vqshrun_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrun_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrun_high_n_s32<const N: i32>(a: uint16x4_t, b: int32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vqshrun_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrun_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrun_high_n_s64<const N: i32>(a: uint32x2_t, b: int64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vqshrun_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrund_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrund_n_s64<const N: i32>(a: i64) -> u32 {
    static_assert!(N >= 1 && N <= 32);
    simd_extract!(vqshrun_n_s64::<N>(vdupq_n_s64(a)), 0)
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshrunh_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshrunh_n_s16<const N: i32>(a: i16) -> u8 {
    static_assert!(N >= 1 && N <= 8);
    simd_extract!(vqshrun_n_s16::<N>(vdupq_n_s16(a)), 0)
}

#[doc = "Signed saturating shift right unsigned narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqshruns_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vqshruns_n_s32<const N: i32>(a: i32) -> u16 {
    static_assert!(N >= 1 && N <= 16);
    simd_extract!(vqshrun_n_s32::<N>(vdupq_n_s32(a)), 0)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubb_s8(a: i8, b: i8) -> i8 {
    let a: int8x8_t = vdup_n_s8(a);
    let b: int8x8_t = vdup_n_s8(b);
    simd_extract!(vqsub_s8(a, b), 0)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubh_s16(a: i16, b: i16) -> i16 {
    let a: int16x4_t = vdup_n_s16(a);
    let b: int16x4_t = vdup_n_s16(b);
    simd_extract!(vqsub_s16(a, b), 0)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubb_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubb_u8(a: u8, b: u8) -> u8 {
    let a: uint8x8_t = vdup_n_u8(a);
    let b: uint8x8_t = vdup_n_u8(b);
    simd_extract!(vqsub_u8(a, b), 0)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubh_u16(a: u16, b: u16) -> u16 {
    let a: uint16x4_t = vdup_n_u16(a);
    let b: uint16x4_t = vdup_n_u16(b);
    simd_extract!(vqsub_u16(a, b), 0)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubs_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubs_s32(a: i32, b: i32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqsub.i32"
        )]
        fn _vqsubs_s32(a: i32, b: i32) -> i32;
    }
    _vqsubs_s32(a, b)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(sqsub))]
pub unsafe fn vqsubd_s64(a: i64, b: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.sqsub.i64"
        )]
        fn _vqsubd_s64(a: i64, b: i64) -> i64;
    }
    _vqsubd_s64(a, b)
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubs_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubs_u32(a: u32, b: u32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqsub.i32"
        )]
        fn _vqsubs_u32(a: i32, b: i32) -> i32;
    }
    _vqsubs_u32(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Saturating subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vqsubd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(uqsub))]
pub unsafe fn vqsubd_u64(a: u64, b: u64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.uqsub.i64"
        )]
        fn _vqsubd_u64(a: i64, b: i64) -> i64;
    }
    _vqsubd_u64(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Rotate and exclusive OR"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrax1q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[cfg_attr(test, assert_instr(rax1))]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
pub unsafe fn vrax1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.rax1"
        )]
        fn _vrax1q_u64(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
    _vrax1q_u64(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbit_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_s8(a: int8x8_t) -> int8x8_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.rbit.v8i8"
        )]
        fn _vrbit_s8(a: int8x8_t) -> int8x8_t;
    }
    _vrbit_s8(a)
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbitq_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_s8(a: int8x16_t) -> int8x16_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.rbit.v16i8"
        )]
        fn _vrbitq_s8(a: int8x16_t) -> int8x16_t;
    }
    _vrbitq_s8(a)
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbit_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_u8(a: uint8x8_t) -> uint8x8_t {
    transmute(vrbit_s8(transmute(a)))
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbitq_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_u8(a: uint8x16_t) -> uint8x16_t {
    transmute(vrbitq_s8(transmute(a)))
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbit_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbit_p8(a: poly8x8_t) -> poly8x8_t {
    transmute(vrbit_s8(transmute(a)))
}

#[doc = "Reverse bit order"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrbitq_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(rbit))]
pub unsafe fn vrbitq_p8(a: poly8x16_t) -> poly8x16_t {
    transmute(vrbitq_s8(transmute(a)))
}

#[doc = "Reciprocal estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpe_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpe_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpe.v1f64"
        )]
        fn _vrecpe_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrecpe_f64(a)
}

#[doc = "Reciprocal estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpeq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpeq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpe.v2f64"
        )]
        fn _vrecpeq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrecpeq_f64(a)
}

#[doc = "Reciprocal estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecped_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecped_f64(a: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpe.f64"
        )]
        fn _vrecped_f64(a: f64) -> f64;
    }
    _vrecped_f64(a)
}

#[doc = "Reciprocal estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpes_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpe))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpes_f32(a: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpe.f32"
        )]
        fn _vrecpes_f32(a: f32) -> f32;
    }
    _vrecpes_f32(a)
}

#[doc = "Floating-point reciprocal step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecps_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecps_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecps.v1f64"
        )]
        fn _vrecps_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vrecps_f64(a, b)
}

#[doc = "Floating-point reciprocal step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpsq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpsq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecps.v2f64"
        )]
        fn _vrecpsq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vrecpsq_f64(a, b)
}

#[doc = "Floating-point reciprocal step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpsd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpsd_f64(a: f64, b: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecps.f64"
        )]
        fn _vrecpsd_f64(a: f64, b: f64) -> f64;
    }
    _vrecpsd_f64(a, b)
}

#[doc = "Floating-point reciprocal step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpss_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecps))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpss_f32(a: f32, b: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecps.f32"
        )]
        fn _vrecpss_f32(a: f32, b: f32) -> f32;
    }
    _vrecpss_f32(a, b)
}

#[doc = "Floating-point reciprocal exponent"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpxd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpx))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpxd_f64(a: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpx.f64"
        )]
        fn _vrecpxd_f64(a: f64) -> f64;
    }
    _vrecpxd_f64(a)
}

#[doc = "Floating-point reciprocal exponent"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrecpxs_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frecpx))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrecpxs_f32(a: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frecpx.f32"
        )]
        fn _vrecpxs_f32(a: f32) -> f32;
    }
    _vrecpxs_f32(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_p128)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_p128(a: p128) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_f32(a: float32x2_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p64_f32(a: float32x2_t) -> poly64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_f32(a: float32x4_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p64_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p64_f32(a: float32x4_t) -> poly64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f32_f64(a: float64x1_t) -> float32x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_s8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_s8_f64(a: float64x1_t) -> int8x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_s16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_s16_f64(a: float64x1_t) -> int16x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_s32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_s32_f64(a: float64x1_t) -> int32x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_s64_f64(a: float64x1_t) -> int64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_u8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u8_f64(a: float64x1_t) -> uint8x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_u16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u16_f64(a: float64x1_t) -> uint16x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_u32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u32_f64(a: float64x1_t) -> uint32x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u64_f64(a: float64x1_t) -> uint64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p8_f64(a: float64x1_t) -> poly8x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p16_f64(a: float64x1_t) -> poly16x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p64_f64(a: float64x1_t) -> poly64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p128_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p128_f64(a: float64x2_t) -> p128 {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f32_f64(a: float64x2_t) -> float32x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_s8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s8_f64(a: float64x2_t) -> int8x16_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_s16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s16_f64(a: float64x2_t) -> int16x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_s32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s32_f64(a: float64x2_t) -> int32x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_s64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s64_f64(a: float64x2_t) -> int64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u8_f64(a: float64x2_t) -> uint8x16_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u16_f64(a: float64x2_t) -> uint16x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u32_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u32_f64(a: float64x2_t) -> uint32x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u64_f64(a: float64x2_t) -> uint64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p8_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p8_f64(a: float64x2_t) -> poly8x16_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p16_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p16_f64(a: float64x2_t) -> poly16x8_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p64_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p64_f64(a: float64x2_t) -> poly64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_s8(a: int8x8_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_s8(a: int8x16_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_s16(a: int16x4_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_s16(a: int16x8_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_s32(a: int32x2_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_s32(a: int32x4_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_s64(a: int64x1_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p64_s64(a: int64x1_t) -> poly64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_s64(a: int64x2_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p64_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p64_s64(a: int64x2_t) -> poly64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_u8(a: uint8x8_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_u8(a: uint8x16_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_u16(a: uint16x4_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_u16(a: uint16x8_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_u32(a: uint32x2_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_u32(a: uint32x4_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_u64(a: uint64x1_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_p64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_p64_u64(a: uint64x1_t) -> poly64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_u64(a: uint64x2_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_p64_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_p64_u64(a: uint64x2_t) -> poly64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_p8(a: poly8x8_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_p8(a: poly8x16_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_p16(a: poly16x4_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_p16(a: poly16x8_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f32_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f32_p64(a: poly64x1_t) -> float32x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_f64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_f64_p64(a: poly64x1_t) -> float64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_s64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_s64_p64(a: poly64x1_t) -> int64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpret_u64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpret_u64_p64(a: poly64x1_t) -> uint64x1_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f32_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f32_p64(a: poly64x2_t) -> float32x4_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_f64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_f64_p64(a: poly64x2_t) -> float64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_s64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_s64_p64(a: poly64x2_t) -> int64x2_t {
    transmute(a)
}

#[doc = "Vector reinterpret cast operation"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vreinterpretq_u64_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vreinterpretq_u64_p64(a: poly64x2_t) -> uint64x2_t {
    transmute(a)
}

#[doc = "Floating-point round to 32-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32x_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32x))]
pub unsafe fn vrnd32x_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32x.v2f32"
        )]
        fn _vrnd32x_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnd32x_f32(a)
}

#[doc = "Floating-point round to 32-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32xq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32x))]
pub unsafe fn vrnd32xq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32x.v4f32"
        )]
        fn _vrnd32xq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrnd32xq_f32(a)
}

#[doc = "Floating-point round to 32-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32xq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32x))]
pub unsafe fn vrnd32xq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32x.v2f64"
        )]
        fn _vrnd32xq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrnd32xq_f64(a)
}

#[doc = "Floating-point round to 32-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32x_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32x))]
pub unsafe fn vrnd32x_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.frint32x.f64"
        )]
        fn _vrnd32x_f64(a: f64) -> f64;
    }
    transmute(_vrnd32x_f64(simd_extract!(a, 0)))
}

#[doc = "Floating-point round to 32-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32z_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32z))]
pub unsafe fn vrnd32z_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32z.v2f32"
        )]
        fn _vrnd32z_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnd32z_f32(a)
}

#[doc = "Floating-point round to 32-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32zq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32z))]
pub unsafe fn vrnd32zq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32z.v4f32"
        )]
        fn _vrnd32zq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrnd32zq_f32(a)
}

#[doc = "Floating-point round to 32-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32zq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32z))]
pub unsafe fn vrnd32zq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint32z.v2f64"
        )]
        fn _vrnd32zq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrnd32zq_f64(a)
}

#[doc = "Floating-point round to 32-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd32z_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint32z))]
pub unsafe fn vrnd32z_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.frint32z.f64"
        )]
        fn _vrnd32z_f64(a: f64) -> f64;
    }
    transmute(_vrnd32z_f64(simd_extract!(a, 0)))
}

#[doc = "Floating-point round to 64-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64x_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64x))]
pub unsafe fn vrnd64x_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64x.v2f32"
        )]
        fn _vrnd64x_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnd64x_f32(a)
}

#[doc = "Floating-point round to 64-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64xq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64x))]
pub unsafe fn vrnd64xq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64x.v4f32"
        )]
        fn _vrnd64xq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrnd64xq_f32(a)
}

#[doc = "Floating-point round to 64-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64xq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64x))]
pub unsafe fn vrnd64xq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64x.v2f64"
        )]
        fn _vrnd64xq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrnd64xq_f64(a)
}

#[doc = "Floating-point round to 64-bit integer, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64x_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64x))]
pub unsafe fn vrnd64x_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.frint64x.f64"
        )]
        fn _vrnd64x_f64(a: f64) -> f64;
    }
    transmute(_vrnd64x_f64(simd_extract!(a, 0)))
}

#[doc = "Floating-point round to 64-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64z_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64z))]
pub unsafe fn vrnd64z_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64z.v2f32"
        )]
        fn _vrnd64z_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnd64z_f32(a)
}

#[doc = "Floating-point round to 64-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64zq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64z))]
pub unsafe fn vrnd64zq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64z.v4f32"
        )]
        fn _vrnd64zq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrnd64zq_f32(a)
}

#[doc = "Floating-point round to 64-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64zq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64z))]
pub unsafe fn vrnd64zq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frint64z.v2f64"
        )]
        fn _vrnd64zq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrnd64zq_f64(a)
}

#[doc = "Floating-point round to 64-bit integer toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd64z_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,frintts")]
#[unstable(feature = "stdarch_neon_ftts", issue = "117227")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(frint64z))]
pub unsafe fn vrnd64z_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.frint64z.f64"
        )]
        fn _vrnd64z_f64(a: f64) -> f64;
    }
    transmute(_vrnd64z_f64(simd_extract!(a, 0)))
}

#[doc = "Floating-point round to integral, toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrnd_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.trunc.v2f32"
        )]
        fn _vrnd_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnd_f32(a)
}

#[doc = "Floating-point round to integral, toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrndq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.trunc.v4f32"
        )]
        fn _vrndq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndq_f32(a)
}

#[doc = "Floating-point round to integral, toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrnd_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.trunc.v1f64"
        )]
        fn _vrnd_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrnd_f64(a)
}

#[doc = "Floating-point round to integral, toward zero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintz))]
pub unsafe fn vrndq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.trunc.v2f64"
        )]
        fn _vrndq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndq_f64(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnda_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrnda_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.round.v2f32"
        )]
        fn _vrnda_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrnda_f32(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndaq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrndaq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.round.v4f32"
        )]
        fn _vrndaq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndaq_f32(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrnda_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrnda_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.round.v1f64"
        )]
        fn _vrnda_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrnda_f64(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to away"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndaq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinta))]
pub unsafe fn vrndaq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.round.v2f64"
        )]
        fn _vrndaq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndaq_f64(a)
}

#[doc = "Floating-point round to integral, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndi_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndi_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.nearbyint.v2f32"
        )]
        fn _vrndi_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrndi_f32(a)
}

#[doc = "Floating-point round to integral, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndiq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndiq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.nearbyint.v4f32"
        )]
        fn _vrndiq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndiq_f32(a)
}

#[doc = "Floating-point round to integral, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndi_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndi_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.nearbyint.v1f64"
        )]
        fn _vrndi_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrndi_f64(a)
}

#[doc = "Floating-point round to integral, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndiq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frinti))]
pub unsafe fn vrndiq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.nearbyint.v2f64"
        )]
        fn _vrndiq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndiq_f64(a)
}

#[doc = "Floating-point round to integral, toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndm_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndm_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.floor.v2f32"
        )]
        fn _vrndm_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrndm_f32(a)
}

#[doc = "Floating-point round to integral, toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndmq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndmq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.floor.v4f32"
        )]
        fn _vrndmq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndmq_f32(a)
}

#[doc = "Floating-point round to integral, toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndm_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndm_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.floor.v1f64"
        )]
        fn _vrndm_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrndm_f64(a)
}

#[doc = "Floating-point round to integral, toward minus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndmq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintm))]
pub unsafe fn vrndmq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.floor.v2f64"
        )]
        fn _vrndmq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndmq_f64(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndn_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintn))]
pub unsafe fn vrndn_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frintn.v1f64"
        )]
        fn _vrndn_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrndn_f64(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndnq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintn))]
pub unsafe fn vrndnq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frintn.v2f64"
        )]
        fn _vrndnq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndnq_f64(a)
}

#[doc = "Floating-point round to integral, to nearest with ties to even"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndns_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintn))]
pub unsafe fn vrndns_f32(a: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.roundeven.f32"
        )]
        fn _vrndns_f32(a: f32) -> f32;
    }
    _vrndns_f32(a)
}

#[doc = "Floating-point round to integral, toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndp_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndp_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.ceil.v2f32"
        )]
        fn _vrndp_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrndp_f32(a)
}

#[doc = "Floating-point round to integral, toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndpq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndpq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.ceil.v4f32"
        )]
        fn _vrndpq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndpq_f32(a)
}

#[doc = "Floating-point round to integral, toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndp_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndp_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.ceil.v1f64"
        )]
        fn _vrndp_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrndp_f64(a)
}

#[doc = "Floating-point round to integral, toward plus infinity"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndpq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintp))]
pub unsafe fn vrndpq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.ceil.v2f64"
        )]
        fn _vrndpq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndpq_f64(a)
}

#[doc = "Floating-point round to integral exact, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndx_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndx_f32(a: float32x2_t) -> float32x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.rint.v2f32"
        )]
        fn _vrndx_f32(a: float32x2_t) -> float32x2_t;
    }
    _vrndx_f32(a)
}

#[doc = "Floating-point round to integral exact, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndxq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndxq_f32(a: float32x4_t) -> float32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.rint.v4f32"
        )]
        fn _vrndxq_f32(a: float32x4_t) -> float32x4_t;
    }
    _vrndxq_f32(a)
}

#[doc = "Floating-point round to integral exact, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndx_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndx_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.rint.v1f64"
        )]
        fn _vrndx_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrndx_f64(a)
}

#[doc = "Floating-point round to integral exact, using current rounding mode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrndxq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(frintx))]
pub unsafe fn vrndxq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.rint.v2f64"
        )]
        fn _vrndxq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrndxq_f64(a)
}

#[doc = "Signed rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshld_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshld_s64(a: i64, b: i64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.srshl.i64"
        )]
        fn _vrshld_s64(a: i64, b: i64) -> i64;
    }
    _vrshld_s64(a, b)
}

#[doc = "Unsigned rounding shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshld_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(urshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshld_u64(a: u64, b: i64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.urshl.i64"
        )]
        fn _vrshld_u64(a: i64, b: i64) -> i64;
    }
    _vrshld_u64(a.as_signed(), b).as_unsigned()
}

#[doc = "Signed rounding shift right"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrd_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrd_n_s64<const N: i32>(a: i64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    vrshld_s64(a, -N as i64)
}

#[doc = "Unsigned rounding shift right"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrd_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrd_n_u64<const N: i32>(a: u64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    vrshld_u64(a, -N as i64)
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vrshrn_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vrshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vrshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vrshrn_n_u16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vrshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Rounding shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrshrn_high_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rshrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vrshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Reciprocal square-root estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrte_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrte_f64(a: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrte.v1f64"
        )]
        fn _vrsqrte_f64(a: float64x1_t) -> float64x1_t;
    }
    _vrsqrte_f64(a)
}

#[doc = "Reciprocal square-root estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrteq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrteq_f64(a: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrte.v2f64"
        )]
        fn _vrsqrteq_f64(a: float64x2_t) -> float64x2_t;
    }
    _vrsqrteq_f64(a)
}

#[doc = "Reciprocal square-root estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrted_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrted_f64(a: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrte.f64"
        )]
        fn _vrsqrted_f64(a: f64) -> f64;
    }
    _vrsqrted_f64(a)
}

#[doc = "Reciprocal square-root estimate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrtes_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrte))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrtes_f32(a: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrte.f32"
        )]
        fn _vrsqrtes_f32(a: f32) -> f32;
    }
    _vrsqrtes_f32(a)
}

#[doc = "Floating-point reciprocal square root step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrts_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrts))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrts_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrts.v1f64"
        )]
        fn _vrsqrts_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t;
    }
    _vrsqrts_f64(a, b)
}

#[doc = "Floating-point reciprocal square root step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrtsq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrts))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrtsq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrts.v2f64"
        )]
        fn _vrsqrtsq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t;
    }
    _vrsqrtsq_f64(a, b)
}

#[doc = "Floating-point reciprocal square root step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrtsd_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrts))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrtsd_f64(a: f64, b: f64) -> f64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrts.f64"
        )]
        fn _vrsqrtsd_f64(a: f64, b: f64) -> f64;
    }
    _vrsqrtsd_f64(a, b)
}

#[doc = "Floating-point reciprocal square root step"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsqrtss_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(frsqrts))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsqrtss_f32(a: f32, b: f32) -> f32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.frsqrts.f32"
        )]
        fn _vrsqrtss_f32(a: f32, b: f32) -> f32;
    }
    _vrsqrtss_f32(a, b)
}

#[doc = "Signed rounding shift right and accumulate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsrad_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsrad_n_s64<const N: i32>(a: i64, b: i64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    let b: i64 = vrshrd_n_s64::<N>(b);
    a.wrapping_add(b)
}

#[doc = "Unsigned rounding shift right and accumulate."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsrad_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsrad_n_u64<const N: i32>(a: u64, b: u64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    let b: u64 = vrshrd_n_u64::<N>(b);
    a.wrapping_add(b)
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_s16(a: int8x8_t, b: int16x8_t, c: int16x8_t) -> int8x16_t {
    let x: int8x8_t = vrsubhn_s16(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_s32(a: int16x4_t, b: int32x4_t, c: int32x4_t) -> int16x8_t {
    let x: int16x4_t = vrsubhn_s32(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_s64(a: int32x2_t, b: int64x2_t, c: int64x2_t) -> int32x4_t {
    let x: int32x2_t = vrsubhn_s64(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3])
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_u16(a: uint8x8_t, b: uint16x8_t, c: uint16x8_t) -> uint8x16_t {
    let x: uint8x8_t = vrsubhn_u16(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_u32(a: uint16x4_t, b: uint32x4_t, c: uint32x4_t) -> uint16x8_t {
    let x: uint16x4_t = vrsubhn_u32(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Rounding subtract returning high narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vrsubhn_high_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(rsubhn2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vrsubhn_high_u64(a: uint32x2_t, b: uint64x2_t, c: uint64x2_t) -> uint32x4_t {
    let x: uint32x2_t = vrsubhn_u64(b, c);
    simd_shuffle!(a, x, [0, 1, 2, 3])
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vset_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vset_lane_f64<const LANE: i32>(a: f64, b: float64x1_t) -> float64x1_t {
    static_assert!(LANE == 0);
    simd_insert!(b, LANE as u32, a)
}

#[doc = "Insert vector element from another vector element"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsetq_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsetq_lane_f64<const LANE: i32>(a: f64, b: float64x2_t) -> float64x2_t {
    static_assert_uimm_bits!(LANE, 1);
    simd_insert!(b, LANE as u32, a)
}

#[doc = "SHA512 hash update part 2"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsha512h2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[cfg_attr(test, assert_instr(sha512h2))]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
pub unsafe fn vsha512h2q_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sha512h2"
        )]
        fn _vsha512h2q_u64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _vsha512h2q_u64(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "SHA512 hash update part 1"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsha512hq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[cfg_attr(test, assert_instr(sha512h))]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
pub unsafe fn vsha512hq_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sha512h"
        )]
        fn _vsha512hq_u64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _vsha512hq_u64(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "SHA512 schedule update 0"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsha512su0q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[cfg_attr(test, assert_instr(sha512su0))]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
pub unsafe fn vsha512su0q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sha512su0"
        )]
        fn _vsha512su0q_u64(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
    _vsha512su0q_u64(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "SHA512 schedule update 1"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsha512su1q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sha3")]
#[cfg_attr(test, assert_instr(sha512su1))]
#[stable(feature = "stdarch_neon_sha3", since = "1.79.0")]
pub unsafe fn vsha512su1q_u64(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sha512su1"
        )]
        fn _vsha512su1q_u64(a: int64x2_t, b: int64x2_t, c: int64x2_t) -> int64x2_t;
    }
    _vsha512su1q_u64(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "Signed Shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshld_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshld_s64(a: i64, b: i64) -> i64 {
    transmute(vshl_s64(transmute(a), transmute(b)))
}

#[doc = "Unsigned Shift left"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshld_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushl))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshld_u64(a: u64, b: i64) -> u64 {
    transmute(vshl_u64(transmute(a), transmute(b)))
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_s8<const N: i32>(a: int8x16_t) -> int16x8_t {
    static_assert!(N >= 0 && N <= 8);
    let b: int8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vshll_n_s8::<N>(b)
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_s16<const N: i32>(a: int16x8_t) -> int32x4_t {
    static_assert!(N >= 0 && N <= 16);
    let b: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    vshll_n_s16::<N>(b)
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(sshll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_s32<const N: i32>(a: int32x4_t) -> int64x2_t {
    static_assert!(N >= 0 && N <= 32);
    let b: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    vshll_n_s32::<N>(b)
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_u8<const N: i32>(a: uint8x16_t) -> uint16x8_t {
    static_assert!(N >= 0 && N <= 8);
    let b: uint8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    vshll_n_u8::<N>(b)
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_u16<const N: i32>(a: uint16x8_t) -> uint32x4_t {
    static_assert!(N >= 0 && N <= 16);
    let b: uint16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    vshll_n_u16::<N>(b)
}

#[doc = "Signed shift left long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshll_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(ushll2, N = 2))]
#[rustc_legacy_const_generics(1)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshll_high_n_u32<const N: i32>(a: uint32x4_t) -> uint64x2_t {
    static_assert!(N >= 0 && N <= 32);
    let b: uint32x2_t = simd_shuffle!(a, a, [2, 3]);
    vshll_n_u32::<N>(b)
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_s16<const N: i32>(a: int8x8_t, b: int16x8_t) -> int8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vshrn_n_s16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_s32<const N: i32>(a: int16x4_t, b: int32x4_t) -> int16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vshrn_n_s32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_s64<const N: i32>(a: int32x2_t, b: int64x2_t) -> int32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vshrn_n_s64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_u16<const N: i32>(a: uint8x8_t, b: uint16x8_t) -> uint8x16_t {
    static_assert!(N >= 1 && N <= 8);
    simd_shuffle!(
        a,
        vshrn_n_u16::<N>(b),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    )
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_u32<const N: i32>(a: uint16x4_t, b: uint32x4_t) -> uint16x8_t {
    static_assert!(N >= 1 && N <= 16);
    simd_shuffle!(a, vshrn_n_u32::<N>(b), [0, 1, 2, 3, 4, 5, 6, 7])
}

#[doc = "Shift right narrow"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vshrn_high_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(shrn2, N = 2))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vshrn_high_n_u64<const N: i32>(a: uint32x2_t, b: uint64x2_t) -> uint32x4_t {
    static_assert!(N >= 1 && N <= 32);
    simd_shuffle!(a, vshrn_n_u64::<N>(b), [0, 1, 2, 3])
}

#[doc = "Shift left and insert"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vslid_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sli, N = 2))]
pub unsafe fn vslid_n_s64<const N: i32>(a: i64, b: i64) -> i64 {
    static_assert!(N >= 0 && N <= 63);
    transmute(vsli_n_s64::<N>(transmute(a), transmute(b)))
}

#[doc = "Shift left and insert"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vslid_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sli, N = 2))]
pub unsafe fn vslid_n_u64<const N: i32>(a: u64, b: u64) -> u64 {
    static_assert!(N >= 0 && N <= 63);
    transmute(vsli_n_u64::<N>(transmute(a), transmute(b)))
}

#[doc = "SM3PARTW1"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsm3partw1q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sm4")]
#[cfg_attr(test, assert_instr(sm3partw1))]
#[unstable(feature = "stdarch_neon_sm4", issue = "117226")]
pub unsafe fn vsm3partw1q_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sm3partw1"
        )]
        fn _vsm3partw1q_u32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vsm3partw1q_u32(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "SM3PARTW2"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsm3partw2q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sm4")]
#[cfg_attr(test, assert_instr(sm3partw2))]
#[unstable(feature = "stdarch_neon_sm4", issue = "117226")]
pub unsafe fn vsm3partw2q_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sm3partw2"
        )]
        fn _vsm3partw2q_u32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vsm3partw2q_u32(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "SM3SS1"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsm3ss1q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sm4")]
#[cfg_attr(test, assert_instr(sm3ss1))]
#[unstable(feature = "stdarch_neon_sm4", issue = "117226")]
pub unsafe fn vsm3ss1q_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sm3ss1"
        )]
        fn _vsm3ss1q_u32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t;
    }
    _vsm3ss1q_u32(a.as_signed(), b.as_signed(), c.as_signed()).as_unsigned()
}

#[doc = "SM4 key"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsm4ekeyq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sm4")]
#[cfg_attr(test, assert_instr(sm4ekey))]
#[unstable(feature = "stdarch_neon_sm4", issue = "117226")]
pub unsafe fn vsm4ekeyq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sm4ekey"
        )]
        fn _vsm4ekeyq_u32(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
    _vsm4ekeyq_u32(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "SM4 encode"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsm4eq_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,sm4")]
#[cfg_attr(test, assert_instr(sm4e))]
#[unstable(feature = "stdarch_neon_sm4", issue = "117226")]
pub unsafe fn vsm4eq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.crypto.sm4e"
        )]
        fn _vsm4eq_u32(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
    _vsm4eq_u32(a.as_signed(), b.as_signed()).as_unsigned()
}

#[doc = "Unsigned saturating accumulate of signed value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqaddb_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqaddb_u8(a: u8, b: i8) -> u8 {
    simd_extract!(vsqadd_u8(vdup_n_u8(a), vdup_n_s8(b)), 0)
}

#[doc = "Unsigned saturating accumulate of signed value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqaddh_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqaddh_u16(a: u16, b: i16) -> u16 {
    simd_extract!(vsqadd_u16(vdup_n_u16(a), vdup_n_s16(b)), 0)
}

#[doc = "Unsigned saturating accumulate of signed value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqaddd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqaddd_u64(a: u64, b: i64) -> u64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.usqadd.i64"
        )]
        fn _vsqaddd_u64(a: i64, b: i64) -> i64;
    }
    _vsqaddd_u64(a.as_signed(), b).as_unsigned()
}

#[doc = "Unsigned saturating accumulate of signed value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqadds_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(usqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqadds_u32(a: u32, b: i32) -> u32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.usqadd.i32"
        )]
        fn _vsqadds_u32(a: i32, b: i32) -> i32;
    }
    _vsqadds_u32(a.as_signed(), b).as_unsigned()
}

#[doc = "Calculates the square root of each lane."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqrt_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqrt_f32(a: float32x2_t) -> float32x2_t {
    simd_fsqrt(a)
}

#[doc = "Calculates the square root of each lane."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqrtq_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqrtq_f32(a: float32x4_t) -> float32x4_t {
    simd_fsqrt(a)
}

#[doc = "Calculates the square root of each lane."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqrt_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqrt_f64(a: float64x1_t) -> float64x1_t {
    simd_fsqrt(a)
}

#[doc = "Calculates the square root of each lane."]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsqrtq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(fsqrt))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vsqrtq_f64(a: float64x2_t) -> float64x2_t {
    simd_fsqrt(a)
}

#[doc = "Shift right and insert"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsrid_n_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sri, N = 2))]
pub unsafe fn vsrid_n_s64<const N: i32>(a: i64, b: i64) -> i64 {
    static_assert!(N >= 1 && N <= 64);
    transmute(vsri_n_s64::<N>(transmute(a), transmute(b)))
}

#[doc = "Shift right and insert"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsrid_n_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[rustc_legacy_const_generics(2)]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(sri, N = 2))]
pub unsafe fn vsrid_n_u64<const N: i32>(a: u64, b: u64) -> u64 {
    static_assert!(N >= 1 && N <= 64);
    transmute(vsri_n_u64::<N>(transmute(a), transmute(b)))
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1_f64_x2)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1_f64_x2(a: *mut f64, b: float64x1x2_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x2.v1f64.p0f64"
        )]
        fn _vst1_f64_x2(a: float64x1_t, b: float64x1_t, ptr: *mut f64);
    }
    _vst1_f64_x2(b.0, b.1, a)
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1q_f64_x2)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1q_f64_x2(a: *mut f64, b: float64x2x2_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x2.v2f64.p0f64"
        )]
        fn _vst1q_f64_x2(a: float64x2_t, b: float64x2_t, ptr: *mut f64);
    }
    _vst1q_f64_x2(b.0, b.1, a)
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1_f64_x3)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1_f64_x3(a: *mut f64, b: float64x1x3_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x3.v1f64.p0f64"
        )]
        fn _vst1_f64_x3(a: float64x1_t, b: float64x1_t, c: float64x1_t, ptr: *mut f64);
    }
    _vst1_f64_x3(b.0, b.1, b.2, a)
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1q_f64_x3)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1q_f64_x3(a: *mut f64, b: float64x2x3_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x3.v2f64.p0f64"
        )]
        fn _vst1q_f64_x3(a: float64x2_t, b: float64x2_t, c: float64x2_t, ptr: *mut f64);
    }
    _vst1q_f64_x3(b.0, b.1, b.2, a)
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1_f64_x4)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1_f64_x4(a: *mut f64, b: float64x1x4_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x4.v1f64.p0f64"
        )]
        fn _vst1_f64_x4(
            a: float64x1_t,
            b: float64x1_t,
            c: float64x1_t,
            d: float64x1_t,
            ptr: *mut f64,
        );
    }
    _vst1_f64_x4(b.0, b.1, b.2, b.3, a)
}

#[doc = "Store multiple single-element structures to one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1q_f64_x4)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st1))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1q_f64_x4(a: *mut f64, b: float64x2x4_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st1x4.v2f64.p0f64"
        )]
        fn _vst1q_f64_x4(
            a: float64x2_t,
            b: float64x2_t,
            c: float64x2_t,
            d: float64x2_t,
            ptr: *mut f64,
        );
    }
    _vst1q_f64_x4(b.0, b.1, b.2, b.3, a)
}

#[doc = "Store multiple single-element structures from one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1_lane_f64<const LANE: i32>(a: *mut f64, b: float64x1_t) {
    static_assert!(LANE == 0);
    *a = simd_extract!(b, LANE as u32);
}

#[doc = "Store multiple single-element structures from one, two, three, or four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst1q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst1q_lane_f64<const LANE: i32>(a: *mut f64, b: float64x2_t) {
    static_assert_uimm_bits!(LANE, 1);
    *a = simd_extract!(b, LANE as u32);
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st1))]
pub unsafe fn vst2_f64(a: *mut f64, b: float64x1x2_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2.v1f64.p0i8"
        )]
        fn _vst2_f64(a: float64x1_t, b: float64x1_t, ptr: *mut i8);
    }
    _vst2_f64(b.0, b.1, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2_lane_f64<const LANE: i32>(a: *mut f64, b: float64x1x2_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2lane.v1f64.p0i8"
        )]
        fn _vst2_lane_f64(a: float64x1_t, b: float64x1_t, n: i64, ptr: *mut i8);
    }
    _vst2_lane_f64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2_lane_s64<const LANE: i32>(a: *mut i64, b: int64x1x2_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2lane.v1i64.p0i8"
        )]
        fn _vst2_lane_s64(a: int64x1_t, b: int64x1_t, n: i64, ptr: *mut i8);
    }
    _vst2_lane_s64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x1x2_t) {
    static_assert!(LANE == 0);
    vst2_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x1x2_t) {
    static_assert!(LANE == 0);
    vst2_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st2))]
pub unsafe fn vst2q_f64(a: *mut f64, b: float64x2x2_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2.v2f64.p0i8"
        )]
        fn _vst2q_f64(a: float64x2_t, b: float64x2_t, ptr: *mut i8);
    }
    _vst2q_f64(b.0, b.1, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st2))]
pub unsafe fn vst2q_s64(a: *mut i64, b: int64x2x2_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2.v2i64.p0i8"
        )]
        fn _vst2q_s64(a: int64x2_t, b: int64x2_t, ptr: *mut i8);
    }
    _vst2q_s64(b.0, b.1, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_f64<const LANE: i32>(a: *mut f64, b: float64x2x2_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2lane.v2f64.p0i8"
        )]
        fn _vst2q_lane_f64(a: float64x2_t, b: float64x2_t, n: i64, ptr: *mut i8);
    }
    _vst2q_lane_f64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_s8<const LANE: i32>(a: *mut i8, b: int8x16x2_t) {
    static_assert_uimm_bits!(LANE, 4);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2lane.v16i8.p0i8"
        )]
        fn _vst2q_lane_s8(a: int8x16_t, b: int8x16_t, n: i64, ptr: *mut i8);
    }
    _vst2q_lane_s8(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_s64<const LANE: i32>(a: *mut i64, b: int64x2x2_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st2lane.v2i64.p0i8"
        )]
        fn _vst2q_lane_s64(a: int64x2_t, b: int64x2_t, n: i64, ptr: *mut i8);
    }
    _vst2q_lane_s64(b.0, b.1, LANE as i64, a as _)
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x2x2_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst2q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_u8<const LANE: i32>(a: *mut u8, b: uint8x16x2_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst2q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x2x2_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst2q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st2, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_lane_p8<const LANE: i32>(a: *mut p8, b: poly8x16x2_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst2q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st2))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst2q_p64(a: *mut p64, b: poly64x2x2_t) {
    vst2q_s64(transmute(a), transmute(b))
}

#[doc = "Store multiple 2-element structures from two registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st2))]
pub unsafe fn vst2q_u64(a: *mut u64, b: uint64x2x2_t) {
    vst2q_s64(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vst3_f64(a: *mut f64, b: float64x1x3_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3.v1f64.p0i8"
        )]
        fn _vst3_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t, ptr: *mut i8);
    }
    _vst3_f64(b.0, b.1, b.2, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst3_lane_f64<const LANE: i32>(a: *mut f64, b: float64x1x3_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3lane.v1f64.p0i8"
        )]
        fn _vst3_lane_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t, n: i64, ptr: *mut i8);
    }
    _vst3_lane_f64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst3_lane_s64<const LANE: i32>(a: *mut i64, b: int64x1x3_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3lane.v1i64.p0i8"
        )]
        fn _vst3_lane_s64(a: int64x1_t, b: int64x1_t, c: int64x1_t, n: i64, ptr: *mut i8);
    }
    _vst3_lane_s64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x1x3_t) {
    static_assert!(LANE == 0);
    vst3_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x1x3_t) {
    static_assert!(LANE == 0);
    vst3_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3))]
pub unsafe fn vst3q_f64(a: *mut f64, b: float64x2x3_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3.v2f64.p0i8"
        )]
        fn _vst3q_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t, ptr: *mut i8);
    }
    _vst3q_f64(b.0, b.1, b.2, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3))]
pub unsafe fn vst3q_s64(a: *mut i64, b: int64x2x3_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3.v2i64.p0i8"
        )]
        fn _vst3q_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t, ptr: *mut i8);
    }
    _vst3q_s64(b.0, b.1, b.2, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst3q_lane_f64<const LANE: i32>(a: *mut f64, b: float64x2x3_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3lane.v2f64.p0i8"
        )]
        fn _vst3q_lane_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t, n: i64, ptr: *mut i8);
    }
    _vst3q_lane_f64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst3q_lane_s8<const LANE: i32>(a: *mut i8, b: int8x16x3_t) {
    static_assert_uimm_bits!(LANE, 4);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3lane.v16i8.p0i8"
        )]
        fn _vst3q_lane_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t, n: i64, ptr: *mut i8);
    }
    _vst3q_lane_s8(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst3q_lane_s64<const LANE: i32>(a: *mut i64, b: int64x2x3_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st3lane.v2i64.p0i8"
        )]
        fn _vst3q_lane_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t, n: i64, ptr: *mut i8);
    }
    _vst3q_lane_s64(b.0, b.1, b.2, LANE as i64, a as _)
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3q_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x2x3_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst3q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3q_lane_u8<const LANE: i32>(a: *mut u8, b: uint8x16x3_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst3q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3q_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x2x3_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst3q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst3q_lane_p8<const LANE: i32>(a: *mut p8, b: poly8x16x3_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst3q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st3))]
pub unsafe fn vst3q_p64(a: *mut p64, b: poly64x2x3_t) {
    vst3q_s64(transmute(a), transmute(b))
}

#[doc = "Store multiple 3-element structures from three registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst3q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st3))]
pub unsafe fn vst3q_u64(a: *mut u64, b: uint64x2x3_t) {
    vst3q_s64(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vst4_f64(a: *mut f64, b: float64x1x4_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4.v1f64.p0i8"
        )]
        fn _vst4_f64(a: float64x1_t, b: float64x1_t, c: float64x1_t, d: float64x1_t, ptr: *mut i8);
    }
    _vst4_f64(b.0, b.1, b.2, b.3, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst4_lane_f64<const LANE: i32>(a: *mut f64, b: float64x1x4_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4lane.v1f64.p0i8"
        )]
        fn _vst4_lane_f64(
            a: float64x1_t,
            b: float64x1_t,
            c: float64x1_t,
            d: float64x1_t,
            n: i64,
            ptr: *mut i8,
        );
    }
    _vst4_lane_f64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst4_lane_s64<const LANE: i32>(a: *mut i64, b: int64x1x4_t) {
    static_assert!(LANE == 0);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4lane.v1i64.p0i8"
        )]
        fn _vst4_lane_s64(
            a: int64x1_t,
            b: int64x1_t,
            c: int64x1_t,
            d: int64x1_t,
            n: i64,
            ptr: *mut i8,
        );
    }
    _vst4_lane_s64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x1x4_t) {
    static_assert!(LANE == 0);
    vst4_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x1x4_t) {
    static_assert!(LANE == 0);
    vst4_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4))]
pub unsafe fn vst4q_f64(a: *mut f64, b: float64x2x4_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4.v2f64.p0i8"
        )]
        fn _vst4q_f64(a: float64x2_t, b: float64x2_t, c: float64x2_t, d: float64x2_t, ptr: *mut i8);
    }
    _vst4q_f64(b.0, b.1, b.2, b.3, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4))]
pub unsafe fn vst4q_s64(a: *mut i64, b: int64x2x4_t) {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4.v2i64.p0i8"
        )]
        fn _vst4q_s64(a: int64x2_t, b: int64x2_t, c: int64x2_t, d: int64x2_t, ptr: *mut i8);
    }
    _vst4q_s64(b.0, b.1, b.2, b.3, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst4q_lane_f64<const LANE: i32>(a: *mut f64, b: float64x2x4_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4lane.v2f64.p0i8"
        )]
        fn _vst4q_lane_f64(
            a: float64x2_t,
            b: float64x2_t,
            c: float64x2_t,
            d: float64x2_t,
            n: i64,
            ptr: *mut i8,
        );
    }
    _vst4q_lane_f64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst4q_lane_s8<const LANE: i32>(a: *mut i8, b: int8x16x4_t) {
    static_assert_uimm_bits!(LANE, 4);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4lane.v16i8.p0i8"
        )]
        fn _vst4q_lane_s8(
            a: int8x16_t,
            b: int8x16_t,
            c: int8x16_t,
            d: int8x16_t,
            n: i64,
            ptr: *mut i8,
        );
    }
    _vst4q_lane_s8(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vst4q_lane_s64<const LANE: i32>(a: *mut i64, b: int64x2x4_t) {
    static_assert_uimm_bits!(LANE, 1);
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.st4lane.v2i64.p0i8"
        )]
        fn _vst4q_lane_s64(
            a: int64x2_t,
            b: int64x2_t,
            c: int64x2_t,
            d: int64x2_t,
            n: i64,
            ptr: *mut i8,
        );
    }
    _vst4q_lane_s64(b.0, b.1, b.2, b.3, LANE as i64, a as _)
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4q_lane_p64<const LANE: i32>(a: *mut p64, b: poly64x2x4_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst4q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4q_lane_u8<const LANE: i32>(a: *mut u8, b: uint8x16x4_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst4q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4q_lane_u64<const LANE: i32>(a: *mut u64, b: uint64x2x4_t) {
    static_assert_uimm_bits!(LANE, 1);
    vst4q_lane_s64::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_lane_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vst4q_lane_p8<const LANE: i32>(a: *mut p8, b: poly8x16x4_t) {
    static_assert_uimm_bits!(LANE, 4);
    vst4q_lane_s8::<LANE>(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(test, assert_instr(st4))]
pub unsafe fn vst4q_p64(a: *mut p64, b: poly64x2x4_t) {
    vst4q_s64(transmute(a), transmute(b))
}

#[doc = "Store multiple 4-element structures from four registers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vst4q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(st4))]
pub unsafe fn vst4q_u64(a: *mut u64, b: uint64x2x4_t) {
    vst4q_s64(transmute(a), transmute(b))
}

#[doc = "Subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsub_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fsub))]
pub unsafe fn vsub_f64(a: float64x1_t, b: float64x1_t) -> float64x1_t {
    simd_sub(a, b)
}

#[doc = "Subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubq_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(fsub))]
pub unsafe fn vsubq_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_sub(a, b)
}

#[doc = "Subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vsubd_s64(a: i64, b: i64) -> i64 {
    a.wrapping_sub(b)
}

#[doc = "Subtract"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(nop))]
pub unsafe fn vsubd_u64(a: u64, b: u64) -> u64 {
    a.wrapping_sub(b)
}

#[doc = "Signed Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s8(a: int8x16_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: int16x8_t = simd_cast(c);
    let e: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: int16x8_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Signed Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let d: int32x4_t = simd_cast(c);
    let e: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let f: int32x4_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Signed Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubl))]
pub unsafe fn vsubl_high_s32(a: int32x4_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle!(a, a, [2, 3]);
    let d: int64x2_t = simd_cast(c);
    let e: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    let f: int64x2_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Unsigned Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u8(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle!(a, a, [8, 9, 10, 11, 12, 13, 14, 15]);
    let d: uint16x8_t = simd_cast(c);
    let e: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    let f: uint16x8_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Unsigned Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u16(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle!(a, a, [4, 5, 6, 7]);
    let d: uint32x4_t = simd_cast(c);
    let e: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    let f: uint32x4_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Unsigned Subtract Long"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubl_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubl))]
pub unsafe fn vsubl_high_u32(a: uint32x4_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle!(a, a, [2, 3]);
    let d: uint64x2_t = simd_cast(c);
    let e: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    let f: uint64x2_t = simd_cast(e);
    simd_sub(d, f)
}

#[doc = "Signed Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s8(a: int16x8_t, b: int8x16_t) -> int16x8_t {
    let c: int8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Signed Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s16(a: int32x4_t, b: int16x8_t) -> int32x4_t {
    let c: int16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Signed Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(ssubw))]
pub unsafe fn vsubw_high_s32(a: int64x2_t, b: int32x4_t) -> int64x2_t {
    let c: int32x2_t = simd_shuffle!(b, b, [2, 3]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Unsigned Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u8(a: uint16x8_t, b: uint8x16_t) -> uint16x8_t {
    let c: uint8x8_t = simd_shuffle!(b, b, [8, 9, 10, 11, 12, 13, 14, 15]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Unsigned Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u16(a: uint32x4_t, b: uint16x8_t) -> uint32x4_t {
    let c: uint16x4_t = simd_shuffle!(b, b, [4, 5, 6, 7]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Unsigned Subtract Wide"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsubw_high_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(test, assert_instr(usubw))]
pub unsafe fn vsubw_high_u32(a: uint64x2_t, b: uint32x4_t) -> uint64x2_t {
    let c: uint32x2_t = simd_shuffle!(b, b, [2, 3]);
    simd_sub(a, simd_cast(c))
}

#[doc = "Dot product index form with signed and unsigned integers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsudot_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,i8mm")]
#[cfg_attr(test, assert_instr(sudot, LANE = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_i8mm", issue = "117223")]
pub unsafe fn vsudot_laneq_s32<const LANE: i32>(
    a: int32x2_t,
    b: int8x8_t,
    c: uint8x16_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: uint32x4_t = transmute(c);
    let c: uint32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vusdot_s32(a, transmute(c), b)
}

#[doc = "Dot product index form with signed and unsigned integers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vsudotq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,i8mm")]
#[cfg_attr(test, assert_instr(sudot, LANE = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_i8mm", issue = "117223")]
pub unsafe fn vsudotq_laneq_s32<const LANE: i32>(
    a: int32x4_t,
    b: int8x16_t,
    c: uint8x16_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: uint32x4_t = transmute(c);
    let c: uint32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vusdotq_s32(a, transmute(c), b)
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vtrn1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [0, 4, 2, 6])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn1q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn1))]
pub unsafe fn vtrn1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [0, 8, 2, 10, 4, 12, 6, 14])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vtrn2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31]
    )
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [1, 5, 3, 7])
}

#[doc = "Transpose vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtrn2q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(trn2))]
pub unsafe fn vtrn2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [1, 9, 3, 11, 5, 13, 7, 15])
}

#[doc = "Signed compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtst_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtst_s64(a: int64x1_t, b: int64x1_t) -> uint64x1_t {
    let c: int64x1_t = simd_and(a, b);
    let d: i64x1 = i64x1::new(0);
    simd_ne(c, transmute(d))
}

#[doc = "Signed compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtstq_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtstq_s64(a: int64x2_t, b: int64x2_t) -> uint64x2_t {
    let c: int64x2_t = simd_and(a, b);
    let d: i64x2 = i64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

#[doc = "Signed compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtst_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtst_p64(a: poly64x1_t, b: poly64x1_t) -> uint64x1_t {
    let c: poly64x1_t = simd_and(a, b);
    let d: i64x1 = i64x1::new(0);
    simd_ne(c, transmute(d))
}

#[doc = "Signed compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtstq_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtstq_p64(a: poly64x2_t, b: poly64x2_t) -> uint64x2_t {
    let c: poly64x2_t = simd_and(a, b);
    let d: i64x2 = i64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

#[doc = "Unsigned compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtst_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtst_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    let c: uint64x1_t = simd_and(a, b);
    let d: u64x1 = u64x1::new(0);
    simd_ne(c, transmute(d))
}

#[doc = "Unsigned compare bitwise Test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtstq_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(cmtst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtstq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    let c: uint64x2_t = simd_and(a, b);
    let d: u64x2 = u64x2::new(0, 0);
    simd_ne(c, transmute(d))
}

#[doc = "Compare bitwise test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtstd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtstd_s64(a: i64, b: i64) -> u64 {
    transmute(vtst_s64(transmute(a), transmute(b)))
}

#[doc = "Compare bitwise test bits nonzero"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vtstd_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(tst))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vtstd_u64(a: u64, b: u64) -> u64 {
    transmute(vtst_u64(transmute(a), transmute(b)))
}

#[doc = "Signed saturating accumulate of unsigned value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuqaddb_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vuqaddb_s8(a: i8, b: u8) -> i8 {
    simd_extract!(vuqadd_s8(vdup_n_s8(a), vdup_n_u8(b)), 0)
}

#[doc = "Signed saturating accumulate of unsigned value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuqaddh_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vuqaddh_s16(a: i16, b: u16) -> i16 {
    simd_extract!(vuqadd_s16(vdup_n_s16(a), vdup_n_u16(b)), 0)
}

#[doc = "Signed saturating accumulate of unsigned value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuqaddd_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vuqaddd_s64(a: i64, b: u64) -> i64 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.suqadd.i64"
        )]
        fn _vuqaddd_s64(a: i64, b: i64) -> i64;
    }
    _vuqaddd_s64(a, b.as_signed())
}

#[doc = "Signed saturating accumulate of unsigned value"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuqadds_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr(suqadd))]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
pub unsafe fn vuqadds_s32(a: i32, b: u32) -> i32 {
    extern "unadjusted" {
        #[cfg_attr(
            any(target_arch = "aarch64", target_arch = "arm64ec"),
            link_name = "llvm.aarch64.neon.suqadd.i32"
        )]
        fn _vuqadds_s32(a: i32, b: i32) -> i32;
    }
    _vuqadds_s32(a, b.as_signed())
}

#[doc = "Dot product index form with unsigned and signed integers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vusdot_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,i8mm")]
#[cfg_attr(test, assert_instr(usdot, LANE = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_i8mm", issue = "117223")]
pub unsafe fn vusdot_laneq_s32<const LANE: i32>(
    a: int32x2_t,
    b: uint8x8_t,
    c: int8x16_t,
) -> int32x2_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = transmute(c);
    let c: int32x2_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32]);
    vusdot_s32(a, b, transmute(c))
}

#[doc = "Dot product index form with unsigned and signed integers"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vusdotq_laneq_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon,i8mm")]
#[cfg_attr(test, assert_instr(usdot, LANE = 3))]
#[rustc_legacy_const_generics(3)]
#[unstable(feature = "stdarch_neon_i8mm", issue = "117223")]
pub unsafe fn vusdotq_laneq_s32<const LANE: i32>(
    a: int32x4_t,
    b: uint8x16_t,
    c: int8x16_t,
) -> int32x4_t {
    static_assert_uimm_bits!(LANE, 2);
    let c: int32x4_t = transmute(c);
    let c: int32x4_t = simd_shuffle!(c, c, [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vusdotq_s32(a, b, transmute(c))
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vuzp1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [0, 2, 4, 6])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp1q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp1))]
pub unsafe fn vuzp1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [0, 2, 4, 6, 8, 10, 12, 14])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vuzp2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    )
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [1, 3, 5, 7])
}

#[doc = "Unzip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vuzp2q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(uzp2))]
pub unsafe fn vuzp2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [1, 3, 5, 7, 9, 11, 13, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [0, 4, 1, 5])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [0, 8, 1, 9, 2, 10, 3, 11])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip1q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip1))]
pub unsafe fn vzip1q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [0, 2])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_f32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_f64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_s8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_shuffle!(
        a,
        b,
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_s16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_s32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_s64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_u8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_shuffle!(
        a,
        b,
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_u16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_u32)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_u64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_shuffle!(a, b, [1, 3])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_p8)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    simd_shuffle!(
        a,
        b,
        [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31]
    )
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2_p16(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    simd_shuffle!(a, b, [2, 6, 3, 7])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_p16)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_p16(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    simd_shuffle!(a, b, [4, 12, 5, 13, 6, 14, 7, 15])
}

#[doc = "Zip vectors"]
#[doc = "[Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/vzip2q_p64)"]
#[doc = "## Safety"]
#[doc = "  * Neon instrinsic unsafe"]
#[inline]
#[target_feature(enable = "neon")]
#[stable(feature = "neon_intrinsics", since = "1.59.0")]
#[cfg_attr(all(test, not(target_env = "msvc")), assert_instr(zip2))]
pub unsafe fn vzip2q_p64(a: poly64x2_t, b: poly64x2_t) -> poly64x2_t {
    simd_shuffle!(a, b, [1, 3])
}
