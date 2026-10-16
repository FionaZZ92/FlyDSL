# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expand packed E2M1 weights to FP8 in registers, including on gfx942.

Each 32-bit word contains eight FP4 values, least-significant nibble first.
fp4_to_fp8x8 leaves scales separate; mxfp4_to_fp8 applies one E8M0 scale
per 32 weights while materializing the FP8 output.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common.tensor_shim import _to_raw as _raw
from kernels.moe.moe_2stage_a16wmix.utils import _fp4_perm
from kernels.moe.mxfp_moe.mxfp4_gemm_common import global_typed_ptr


def fp4_to_fp8x8(raw_i32, *, fnuz=True):
    """Eight packed E2M1 codes -> eight FP8 codes, preserving element order.

    Scale is intentionally separate. The conversion is exact for all 16
    nibbles, except FP4 -0 becomes +0 in FNUZ (0x80 means NaN in FNUZ).
    Only integer/byte-permute instructions are used, including on gfx942.
    """
    raw = fx.Int32(raw_i32)
    lo = 0x44403800 if fnuz else 0x3C383000  # E4M3FNUZ(gfx942) or E4M3FN
    hi = 0x54504C48 if fnuz else 0x4C484440  # E4M3FNUZ(gfx942) or E4M3FN
    even_sel = raw & fx.Int32(0x07070707)  # x0,x2,x4,x6
    odd_sel = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)  # x1,x3,x5,x7
    even = _fp4_perm(hi, lo, even_sel)
    odd = _fp4_perm(hi, lo, odd_sel)
    sign_even = (raw & fx.Int32(0x08080808)) << fx.Int32(4)
    sign_odd = raw & fx.Int32(0x80808080)
    if const_expr(fnuz):
        # Per-byte nonzero magnitude -> bit7; clear sign on both zero codes.
        nz_even = (even_sel | (even_sel << fx.Int32(1)) | (even_sel << fx.Int32(2))) & fx.Int32(0x04040404)
        nz_odd = (odd_sel | (odd_sel << fx.Int32(1)) | (odd_sel << fx.Int32(2))) & fx.Int32(0x04040404)
        sign_even = sign_even & (nz_even << fx.Int32(5))
        sign_odd = sign_odd & (nz_odd << fx.Int32(5))
    even = even | sign_even
    odd = odd | sign_odd
    p0 = _fp4_perm(odd, even, fx.Int32(0x05010400))  # 0,4,1,5
    p1 = _fp4_perm(odd, even, fx.Int32(0x07030602))  # 2,6,3,7
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    return fx.Vector.from_elements([_raw(p0), _raw(p1)], fx.Int32).bitcast(dtype)


def _scaled_fp8_tables(scale_code, *, fnuz):
    """Build an eight-byte magnitude LUT for E2M1 * 2**(scale_code-127).

    E8M0 scaling shifts the FP8 exponent exactly. Only subnormal outputs
    need rounding. Building the LUT once per 32-value block reuses this work for
    all 32 weights and avoids unsupported generic FP8 casts on gfx942.
    """
    code = fx.Int32(scale_code) & fx.Int32(255)
    delta = code - fx.Int32(127)
    nan_code = 128 if fnuz else 127
    invalid = code == fx.Int32(255)
    lut = [invalid.select(fx.Int32(nan_code), fx.Int32(0))]
    for magnitude in (56, 64, 68, 72, 76, 80, 84) if fnuz else (48, 56, 60, 64, 68, 72, 76):
        exponent = fx.Int32(magnitude >> 3) + delta
        normal = fx.Int32(magnitude) + delta * fx.Int32(8)
        # A subnormal is the four-bit significand shifted by 1-exponent.
        # Clamp before shifting: shifts >= 5 already round every value to 0.
        shift = fx.Int32(1) - exponent
        shift = (shift < fx.Int32(1)).select(fx.Int32(1), shift)
        shift = (shift > fx.Int32(5)).select(fx.Int32(5), shift)
        significand = fx.Int32(8 | (magnitude & 7))
        halfway = fx.Int32(1) << (shift - fx.Int32(1))
        tie_bit = significand.shrui(shift) & fx.Int32(1)
        subnormal = (significand + halfway - fx.Int32(1) + tie_bit).shrui(shift)
        value = (exponent <= fx.Int32(0)).select(subnormal, normal)
        overflow = (normal > fx.Int32(127 if fnuz else 126)) | invalid
        lut.append(overflow.select(fx.Int32(nan_code), value))
    packed = []
    for word in range(2):
        value = lut[word * 4]
        for j in range(1, 4):
            value = value | (lut[word * 4 + j] << fx.Int32(j * 8))
        packed.append(value)
    return packed[0], packed[1], invalid


def _scaled_fp4_to_fp8x8(raw_i32, lo, hi, invalid_scale, *, fnuz):
    raw = fx.Int32(raw_i32)
    even = _fp4_perm(hi, lo, raw & fx.Int32(0x07070707))
    odd = _fp4_perm(hi, lo, raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707))
    sign_even = (raw & fx.Int32(0x08080808)) << fx.Int32(4)
    sign_odd = raw & fx.Int32(0x80808080)
    if const_expr(fnuz):
        # Bit 7 survives iff the magnitude is nonzero. No cross-byte borrow.
        # Underflowed negative values become +0; a LUT NaN remains 0x80.
        nz_even = (even | fx.Int32(0x80808080)) - fx.Int32(0x01010101)
        nz_odd = (odd | fx.Int32(0x80808080)) - fx.Int32(0x01010101)
        sign_even = sign_even & nz_even
        sign_odd = sign_odd & nz_odd
    else:
        # E8M0 NaN has no sign. Finite scales preserve FN signed zero/NaN.
        sign_even = invalid_scale.select(fx.Int32(0), sign_even)
        sign_odd = invalid_scale.select(fx.Int32(0), sign_odd)
    even = even | sign_even
    odd = odd | sign_odd
    p0 = _fp4_perm(odd, even, fx.Int32(0x05010400))
    p1 = _fp4_perm(odd, even, fx.Int32(0x07030602))
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    return fx.Vector.from_elements([_raw(p0), _raw(p1)], fx.Int32).bitcast(dtype)


@functools.cache
def compile_mxfp4_to_fp8(*, groups, fnuz=True):
    """Compile contiguous weight conversion; one thread owns one 32-value block."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    threads = 256

    @flyc.kernel
    def convert(src: fx.Int64, scales: fx.Int64, dst: fx.Int64):
        group = fx.Int64(gpu.block_id("x")) * fx.Int64(threads) + fx.Int64(gpu.thread_id("x"))
        if group < fx.Int64(groups):
            sp = global_typed_ptr(src, T.i32)
            dp = global_typed_ptr(dst, T.i32)
            lo, hi, invalid = _scaled_fp8_tables(global_typed_ptr(scales, T.i8, align=1)[group], fnuz=fnuz)
            packed = fx.Vector(fx.ptr_load(fx.add_offset(sp, group * fx.Int64(4)), result_type=T.vec(4, T.i32)))
            output = []
            for j in range_constexpr(4):
                converted = _scaled_fp4_to_fp8x8(packed[j], lo, hi, invalid, fnuz=fnuz).bitcast(fx.Int32)
                output.extend([_raw(converted[0]), _raw(converted[1])])
            fx.ptr_store(fx.Vector.from_elements(output, fx.Int32), fx.add_offset(dp, group * fx.Int64(8)))

    @flyc.jit
    def launch(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, stream: fx.Stream):
        convert(src, scales, dst).launch(
            grid=((groups + threads - 1) // threads, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


def mxfp4_to_fp8(packed_w, scales, *, out=None):
    """Materialize cast_fp8(E2M1 * E8M0) for contiguous weights [E,N,K/2].

    Inputs are uint8 GPU tensors; scales has shape [E,N,K/32], with K a
    positive multiple of 32. E8M0 codes 0..254 mean 2**(code-127); 255 is
    NaN. Output is [E,N,K], E4M3FNUZ on gfx942 or E4M3FN on gfx950.
    Round to nearest, ties to even; overflow produces NaN, without clamping
    or output rescaling. This is weight conversion only, without GEMM.
    """
    if packed_w.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("packed_w and scales must have dtype uint8")
    if not torch.version.hip or not packed_w.is_cuda or scales.device != packed_w.device:
        raise ValueError("packed_w and scales must be on the same ROCm GPU")
    if not packed_w.is_contiguous() or not scales.is_contiguous():
        raise ValueError("packed_w and scales must be contiguous")
    if packed_w.ndim != 3 or min(packed_w.shape) <= 0 or packed_w.shape[-1] % 16:
        raise ValueError("Expected packed_w[E,N,K/2] with positive dimensions and K divisible by 32")
    e, n, half_k = packed_w.shape
    if tuple(scales.shape) != (e, n, half_k // 16):
        raise ValueError("Expected scales[E,N,K/32]")
    if packed_w.data_ptr() % 4:
        raise ValueError("packed_w must be aligned to four bytes")
    arch = torch.cuda.get_device_properties(packed_w.device).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        raise ValueError("Requires gfx942 or gfx950")
    fnuz = arch == "gfx942"
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    shape = (e, n, half_k * 2)
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=packed_w.device)
    if out.shape != shape or out.dtype != dtype or out.device != packed_w.device or not out.is_contiguous():
        raise ValueError("out must be contiguous FP8 with the expected shape, dtype and device")
    if out.data_ptr() % 4 or any(
        out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in (packed_w, scales)
    ):
        raise ValueError("out must be aligned to four bytes and must not alias the inputs")
    with torch.cuda.device(packed_w.device):
        launch = compile_mxfp4_to_fp8(groups=scales.numel(), fnuz=fnuz)
        launch(packed_w.data_ptr(), scales.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    return out
