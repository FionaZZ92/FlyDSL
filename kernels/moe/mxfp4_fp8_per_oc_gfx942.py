# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Convert MXFP4 to FP8 with a separate power-of-two scale per output channel."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common.tensor_shim import _to_raw as _raw
from kernels.moe.mxfp4_fp8_gfx942 import _scaled_fp4_to_fp8x8, _scaled_fp8_tables
from kernels.moe.mxfp_moe.mxfp4_gemm_common import global_typed_ptr


@functools.cache
def compile_mxfp4_to_fp8_per_oc(*, channels, groups_per_channel, fnuz=True):
    """One wave per channel: reduce E8M0 codes, then convert its K=32 blocks."""
    if channels <= 0 or groups_per_channel <= 0:
        raise ValueError("channels and groups_per_channel must be positive")
    threads = 64
    chunks = (groups_per_channel + threads - 1) // threads
    headroom = 5 if fnuz else 6

    @flyc.kernel
    def convert(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, oc_scales: fx.Int64):
        channel = fx.Int64(gpu.block_id("x"))
        lane = fx.Int32(gpu.thread_id("x"))
        base = channel * fx.Int64(groups_per_channel)
        sp = global_typed_ptr(scales, T.i8, align=1)
        wp = global_typed_ptr(src, T.i32)
        dp = global_typed_ptr(dst, T.i32)
        max_code = fx.Int32(0)
        for chunk in range(chunks):
            local = fx.Int64(chunk * threads) + fx.Int64(lane)
            if local < fx.Int64(groups_per_channel):
                code = fx.Int32(sp[base + local]) & fx.Int32(255)
                # NaN scales affect their own blocks, not finite neighbors.
                finite_code = (code == fx.Int32(255)).select(fx.Int32(0), code)
                max_code = (finite_code > max_code).select(finite_code, max_code)
        for offset in (1, 2, 4, 8, 16, 32):
            other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw((lane ^ fx.Int32(offset)) * fx.Int32(4)), _raw(max_code)))
            max_code = (other > max_code).select(other, max_code)

        # ceil(log2(6 * 2**(max_code-127) / FP8_MAX)) simplifies to
        # max_code-127-headroom. Never materialize the possibly huge bound.
        exponent = max_code - fx.Int32(127 + headroom)
        # Keep the returned FP32 scale normal, including all-tiny channels.
        # Raising this lower bound only decreases normalized weight magnitudes.
        exponent = (exponent < fx.Int32(-126)).select(fx.Int32(-126), exponent)
        if lane == fx.Int32(0):
            scale = ((exponent + fx.Int32(127)) << fx.Int32(23)).bitcast(fx.Float32)
            fx.ptr_store(scale, fx.add_offset(global_typed_ptr(oc_scales, T.f32), channel))

        for chunk in range(chunks):
            local = fx.Int64(chunk * threads) + fx.Int64(lane)
            if local < fx.Int64(groups_per_channel):
                group = base + local
                lo, hi, invalid = _scaled_fp8_tables(sp[group], fnuz=fnuz, exponent_offset=exponent)
                packed = fx.Vector(fx.ptr_load(fx.add_offset(wp, group * fx.Int64(4)), result_type=T.vec(4, T.i32)))
                output = []
                for j in range_constexpr(4):
                    converted = _scaled_fp4_to_fp8x8(packed[j], lo, hi, invalid, fnuz=fnuz).bitcast(fx.Int32)
                    output.extend([_raw(converted[0]), _raw(converted[1])])
                fx.ptr_store(fx.Vector.from_elements(output, fx.Int32), fx.add_offset(dp, group * fx.Int64(8)))

    @flyc.jit
    def launch(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, oc_scales: fx.Int64, stream: fx.Stream):
        convert(src, scales, dst, oc_scales).launch(grid=(channels, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def mxfp4_to_fp8_per_oc(packed_w, scales, *, out=None, out_scale=None):
    """Return (FP8 weights [E,N,K], FP32 dequantization scales [E,N]).

    Inputs are contiguous ROCm uint8 packed E2M1 [E,N,K/2] and E8M0
    [E,N,K/32], with positive dimensions and K divisible by 32. gfx942
    produces E4M3FNUZ; gfx950 produces E4M3FN.

    For each channel, use the conservative bound B=6*max(finite block scales)
    and S=max(2**ceil(log2(B/FP8_MAX)), 2**-126). The kernel computes this
    in exponent space, so neither B nor unnormalized weights need fit FP32.
    Output Q=round_fp8(W/S) reconstructs W approximately as Q*S[...,None].
    Apply S to the corresponding GEMM output channel before bias/activation.

    Every finite input produces finite FP8 output (max magnitude 192 on
    gfx942, 384 on gfx950). Small blocks may still round to zero when scales
    within a channel vary widely. Rounding is nearest-even; FNUZ has no -0.
    E8M0 code 255 produces NaN only in its own 32-value block and is ignored
    by the channel maximum. An all-NaN channel returns S=2**-126 and all NaNs.
    Returned scales are positive, finite, normal FP32 values. Reconstructing
    extremely large weights or GEMM results in FP32 can still overflow.

    Optional out/out_scale must be contiguous, correctly typed/shaped, aligned
    to four bytes, and must not share storage with each other or either input.
    Preallocate both outputs for allocation-free CUDA/HIP graph replay.
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
    if out_scale is None:
        out_scale = torch.empty((e, n), dtype=torch.float32, device=packed_w.device)
    for name, tensor, expected_shape, expected_dtype in (
        ("out", out, shape, dtype),
        ("out_scale", out_scale, (e, n), torch.float32),
    ):
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtype
            or tensor.device != packed_w.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous with the expected shape, dtype and device")
        others = (packed_w, scales, out_scale if name == "out" else out)
        if tensor.data_ptr() % 4 or any(
            tensor.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in others
        ):
            raise ValueError(f"{name} must be aligned to four bytes and must not alias inputs or the other output")
    with torch.cuda.device(packed_w.device):
        launch = compile_mxfp4_to_fp8_per_oc(channels=e * n, groups_per_channel=half_k // 16, fnuz=fnuz)
        launch(
            packed_w.data_ptr(), scales.data_ptr(), out.data_ptr(), out_scale.data_ptr(), torch.cuda.current_stream()
        )
    return out, out_scale
