# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Convert MXFP4 to FP8 with a power-of-two scale per 128x128 weight tile."""

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
def compile_mxfp4_to_fp8_128block(*, experts, n, k, fnuz=True):
    """Reduce tile scales, then convert contiguous K=32 groups in a second pass."""
    if min(experts, n, k) <= 0 or k % 32:
        raise ValueError("experts, n and k must be positive, with K divisible by 32")
    threads = 64
    groups_per_channel = k // 32
    n_tiles, k_tiles = (n + 127) // 128, (k + 127) // 128
    headroom = 5 if fnuz else 6

    @flyc.kernel
    def reduce_scales(scales: fx.Int64, tile_scales: fx.Int64):
        tile = fx.Int64(gpu.block_id("x"))
        lane = fx.Int32(gpu.thread_id("x"))
        expert = tile // fx.Int64(n_tiles * k_tiles)
        row_base = (tile // fx.Int64(k_tiles) % fx.Int64(n_tiles)) * fx.Int64(128)
        group_base = (tile % fx.Int64(k_tiles)) * fx.Int64(4)
        sp = global_typed_ptr(scales, T.i8, align=1)
        max_code = fx.Int32(0)
        for chunk in range(8):
            local = fx.Int64(chunk * threads) + fx.Int64(lane)
            row = row_base + local // fx.Int64(4)
            col = group_base + local % fx.Int64(4)
            if (row < fx.Int64(n)) & (col < fx.Int64(groups_per_channel)):
                group = (expert * fx.Int64(n) + row) * fx.Int64(groups_per_channel) + col
                code = fx.Int32(sp[group]) & fx.Int32(255)
                finite_code = (code == fx.Int32(255)).select(fx.Int32(0), code)
                max_code = (finite_code > max_code).select(finite_code, max_code)
        for offset in (1, 2, 4, 8, 16, 32):
            other = fx.Int32(rocdl.ds_bpermute(T.i32, _raw((lane ^ fx.Int32(offset)) * fx.Int32(4)), _raw(max_code)))
            max_code = (other > max_code).select(other, max_code)
        exponent = max_code - fx.Int32(127 + headroom)
        exponent = (exponent < fx.Int32(-126)).select(fx.Int32(-126), exponent)
        if lane == fx.Int32(0):
            scale = ((exponent + fx.Int32(127)) << fx.Int32(23)).bitcast(fx.Float32)
            fx.ptr_store(scale, fx.add_offset(global_typed_ptr(tile_scales, T.f32), tile))

    @flyc.kernel
    def convert(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, tile_scales: fx.Int64):
        group = fx.Int64(gpu.block_id("x")) * fx.Int64(256) + fx.Int64(gpu.thread_id("x"))
        if group < fx.Int64(experts * n * groups_per_channel):
            channel = group // fx.Int64(groups_per_channel)
            tile = channel // fx.Int64(n) * fx.Int64(n_tiles) + channel % fx.Int64(n) // fx.Int64(128)
            tile = tile * fx.Int64(k_tiles) + group % fx.Int64(groups_per_channel) // fx.Int64(4)
            scale = fx.Float32(global_typed_ptr(tile_scales, T.f32)[tile])
            exponent = (scale.bitcast(fx.Int32) >> fx.Int32(23)) - fx.Int32(127)
            lo, hi, invalid = _scaled_fp8_tables(
                global_typed_ptr(scales, T.i8, align=1)[group], fnuz=fnuz, exponent_offset=exponent
            )
            wp = global_typed_ptr(src, T.i32)
            dp = global_typed_ptr(dst, T.i32)
            packed = fx.Vector(fx.ptr_load(fx.add_offset(wp, group * fx.Int64(4)), result_type=T.vec(4, T.i32)))
            output = []
            for j in range_constexpr(4):
                converted = _scaled_fp4_to_fp8x8(packed[j], lo, hi, invalid, fnuz=fnuz).bitcast(fx.Int32)
                output.extend([_raw(converted[0]), _raw(converted[1])])
            fx.ptr_store(fx.Vector.from_elements(output, fx.Int32), fx.add_offset(dp, group * fx.Int64(8)))

    @flyc.jit
    def launch(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, tile_scales: fx.Int64, stream: fx.Stream):
        reduce_scales(scales, tile_scales).launch(
            grid=(experts * n_tiles * k_tiles, 1, 1), block=(threads, 1, 1), stream=stream
        )
        convert(src, scales, dst, tile_scales).launch(
            grid=((experts * n * groups_per_channel + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


def mxfp4_to_fp8_128block(packed_w, scales, *, out=None, out_scale=None):
    """Return FP8 Q[E,N,K] and FP32 S[E,ceil(N/128),ceil(K/128)].

    Contiguous ROCm uint8 inputs: packed E2M1 [E,N,K/2] and E8M0
    [E,N,K/32], positive dimensions, K divisible by 32. Partial N/K tiles
    use only valid weights. gfx942 produces E4M3FNUZ; gfx950 E4M3FN.

    Like per_oc, use the conservative bound B=6*max(finite E8M0 scales),
    now over each 128x128 tile, and S=max(2**ceil(log2(B/FP8_MAX)), 2**-126).
    This is a bound from input scales, not the actual weight amax. Exponent
    arithmetic avoids FP32 overflow. Q=round_fp8(W/S); reconstruct using
    Q[e,n,k]*S[e,n//128,k//128]. In GEMM, rescale each K tile's partial sum
    before accumulation; S cannot be applied once after the complete dot.

    Finite inputs produce finite FP8 (magnitude <=192 FNUZ / <=384 FN).
    Tiny relative values may round to zero. Rounding is nearest-even.
    E8M0 255 produces NaNs only in its own K=32 group and is excluded from
    the maximum. All-NaN tiles return S=2**-126. Scales are normal finite
    positive FP32; reconstructing extreme weights can still overflow FP32.

    Optional outputs must have the documented shapes, dtype and device,
    be contiguous, four-byte aligned, and share no storage with inputs or
    each other. Preallocate both outputs for allocation-free graph replay.
    The launcher runs scale reduction and contiguous conversion on the same
    stream; out_scale is also the intermediate buffer, with no extra scratch.
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
    scale_shape = (e, (n + 127) // 128, (half_k * 2 + 127) // 128)
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=packed_w.device)
    if out_scale is None:
        out_scale = torch.empty(scale_shape, dtype=torch.float32, device=packed_w.device)
    for name, tensor, expected_shape, expected_dtype in (
        ("out", out, shape, dtype),
        ("out_scale", out_scale, scale_shape, torch.float32),
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
        launch = compile_mxfp4_to_fp8_128block(experts=e, n=n, k=half_k * 2, fnuz=fnuz)
        launch(
            packed_w.data_ptr(), scales.data_ptr(), out.data_ptr(), out_scale.data_ptr(), torch.cuda.current_stream()
        )
    return out, out_scale
