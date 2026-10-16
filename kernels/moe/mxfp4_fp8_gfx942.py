# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Experimental MXFP4-storage / FP8-compute MoE projections for gfx942.

W is packed E2M1 [experts, N, K/2], with uint8 E8M0 [experts, N, K/32].
FP4 codes are expanded directly to FP8 in registers. Each K=32 partial dot
product is multiplied by its block scale before accumulation; weights are
never materialized as a full FP8 tensor. BF16 activations are dynamically
quantized per row and K=32 block. This is a correctness prototype, not a
pipelined/tuned replacement for moe_2stage_a16wmix.

gfx942 uses E4M3FNUZ. gfx950 uses E4M3FN for testing the same algorithm;
successful gfx950 execution does not validate gfx942 machine code.
"""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from kernels.common.tensor_shim import _run_compiled
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
    lo = 0x44403800 if fnuz else 0x3C383000
    hi = 0x54504C48 if fnuz else 0x4C484440
    even_sel = raw & fx.Int32(0x07070707)
    odd_sel = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)
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
    p0 = _fp4_perm(odd, even, fx.Int32(0x05010400))
    p1 = _fp4_perm(odd, even, fx.Int32(0x07030602))
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    return fx.Vector.from_elements([_raw(p0), _raw(p1)], fx.Int32).bitcast(dtype)


def _shuffle_f32(value, source_lane):
    return fx.Int32(rocdl.ds_bpermute(T.i32, _raw(source_lane * fx.Int32(4)), _raw(value.bitcast(fx.Int32)))).bitcast(
        fx.Float32
    )


def _quantize_a(a8, lane, *, fnuz):
    values = [fx.Float32(a8[j]) for j in range_constexpr(8)]
    amax = fx.absf(values[0])
    for j in range_constexpr(1, 8):
        amax = fx.maxnumf(amax, fx.absf(values[j]))
    for offset in (16, 32):
        amax = fx.maxnumf(amax, _shuffle_f32(amax, lane ^ fx.Int32(offset)))
    # Round up to a power-of-two scale, leaving headroom below 240/448.
    # Exact binary rescaling avoids division error at FP8 rounding ties.
    target = fx.maxnumf(amax * fx.Float32(2.0 ** (-7 if fnuz else -8)), fx.Float32(2.0**-120))
    scale_bits = (target.bitcast(fx.Int32) + fx.Int32(0x007FFFFF)) & fx.Int32(0x7F800000)
    scale = scale_bits.bitcast(fx.Float32)
    reciprocal = (fx.Int32(0x7F000000) - scale_bits).bitcast(fx.Float32)
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    normalized = [v * reciprocal for v in values]
    packed = []
    for word in range_constexpr(2):
        i = word * 4
        p = rocdl.cvt_pk_fp8_f32(T.i32, _raw(normalized[i]), _raw(normalized[i + 1]), _raw(fx.Int32(0)), False)
        p = rocdl.cvt_pk_fp8_f32(T.i32, _raw(normalized[i + 2]), _raw(normalized[i + 3]), p, True)
        packed.append(p)
    quant = fx.Vector.from_elements(packed, fx.Int32).bitcast(dtype)
    return quant, scale


def _e8m0_scale(byte):
    b = fx.Int32(byte) & fx.Int32(255)
    normal = (b << fx.Int32(23)).bitcast(fx.Float32)
    finite = (b == fx.Int32(0)).select(fx.Float32(2.0**-127), normal)
    return (b == fx.Int32(255)).select(fx.Int32(0x7FC00000).bitcast(fx.Float32), finite)


def _fragment(v, dtype):
    frag = fx.make_rmem_tensor(fx.make_layout(8, 1), dtype)
    frag.store(v)
    return frag


@functools.cache
def compile_mxfp4_fp8_moe_gemm(*, K, N, experts, a_rows, sorted_rows, gate_up=False, arch="gfx942"):
    """Build a sorted 16-row-tile grouped GEMM, optionally fusing SiLU(gate)*up.

    row_ids maps each sorted row to A; -1 (or any out-of-range row) is padding.
    expert_ids has one expert ID per 16 sorted rows. Invalid experts produce
    zero. Outputs remain in sorted order, including for the down projection.
    Gate/up weights are [gate rows | up rows]. Input weights are contiguous,
    not the preshuffled format used by moe_2stage_a16wmix.
    """
    if arch not in ("gfx942", "gfx950"):
        raise ValueError("This prototype supports gfx942 and gfx950 only")
    if min(K, N, experts, a_rows, sorted_rows) <= 0:
        raise ValueError("All dimensions must be positive")
    if K % 32 or N % (32 if gate_up else 16) or sorted_rows % 16:
        raise ValueError("K must divide by 32; N by 16 (32 for gate/up); sorted_rows by 16")
    fnuz = arch == "gfx942"
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    out_n = N // 2 if gate_up else N
    nblocks = out_n // 16
    projections = 2 if gate_up else 1

    @flyc.kernel(name=f"mxfp4_fp8_moe_{arch}_k{K}_n{N}_gu{int(gate_up)}", known_block_size=[64, 1, 1])
    def kernel(
        arg_a: fx.Int64,
        arg_w: fx.Int64,
        arg_scales: fx.Int64,
        arg_rows: fx.Int64,
        arg_experts: fx.Int64,
        arg_out: fx.Int64,
    ):
        lane = fx.Int32(gpu.thread_id("x"))
        group = lane // fx.Int32(16)
        index = lane % fx.Int32(16)
        bid = fx.Int32(gpu.block_id("x"))
        mblock = bid // fx.Int32(nblocks)
        nbase = (bid % fx.Int32(nblocks)) * fx.Int32(16)
        row_id = fx.Int32(global_typed_ptr(arg_rows, T.i32)[mblock * fx.Int32(16) + index])
        expert = fx.Int32(global_typed_ptr(arg_experts, T.i32)[mblock])
        valid_row = (row_id >= fx.Int32(0)) & (row_id < fx.Int32(a_rows))
        valid_expert = (expert >= fx.Int32(0)) & (expert < fx.Int32(experts))
        safe_row = valid_row.select(row_id, fx.Int32(0))
        safe_expert = valid_expert.select(expert, fx.Int32(0))
        ap = global_typed_ptr(arg_a, T.bf16, align=2)
        wp = global_typed_ptr(arg_w, T.i32)
        sp = global_typed_ptr(arg_scales, T.i8, align=1)
        op = global_typed_ptr(arg_out, T.bf16, align=2)
        atom = fx.make_mma_atom(rocdl.MFMA(16, 16, 32, dtype))
        acc = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(projections)]
        zero = fx.Vector.filled(4, 0.0, fx.Float32)
        for p in range_constexpr(projections):
            acc[p].store(zero)
        for kb in range(K // 32):
            avals = []
            for j in range_constexpr(8):
                offset = fx.Int64(safe_row) * fx.Int64(K) + fx.Int64(kb * 32 + group * 8 + j)
                value = fx.Float32(ap[offset])
                avals.append(valid_row.select(value, fx.Float32(0.0)))
            a8, a_scale = _quantize_a(fx.Vector.from_elements([_raw(v) for v in avals], fx.Float32), lane, fnuz=fnuz)
            afrag = _fragment(a8, dtype)
            # MFMA output element i belongs to row group*4+i and column index.
            out_ascales = [_shuffle_f32(a_scale, group * fx.Int32(4) + fx.Int32(i)) for i in range_constexpr(4)]
            for p in range_constexpr(projections):
                col = nbase + index + fx.Int32(p * out_n)
                wrow = fx.Int64(safe_expert) * fx.Int64(N) + fx.Int64(col)
                packed = fx.Int32(wp[wrow * fx.Int64(K // 8) + fx.Int64(kb * 4 + group)])
                b8 = fp4_to_fp8x8(packed, fnuz=fnuz)
                b_scale = _e8m0_scale(sp[wrow * fx.Int64(K // 32) + fx.Int64(kb)])
                partial = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                partial.store(zero)
                fx.gemm(atom, partial, afrag, _fragment(b8, dtype), partial)
                old, dot = acc[p].load(), partial.load()
                result = [old[i] + dot[i] * out_ascales[i] * b_scale for i in range_constexpr(4)]
                acc[p].store(fx.Vector.from_elements([_raw(v) for v in result], fx.Float32))
        gate = acc[0].load()
        if const_expr(gate_up):
            up = acc[1].load()
        for i in range_constexpr(4):
            row = mblock * fx.Int32(16) + group * fx.Int32(4) + fx.Int32(i)
            output_row_id = fx.Int32(global_typed_ptr(arg_rows, T.i32)[row])
            valid = valid_expert & (output_row_id >= fx.Int32(0)) & (output_row_id < fx.Int32(a_rows))
            value = gate[i]
            if const_expr(gate_up):
                value = (value / (fx.Float32(1.0) + math.exp(-value))) * up[i]
            value = valid.select(value, fx.Float32(0.0))
            fx.ptr_store(
                value.to(fx.BFloat16), fx.add_offset(op, fx.Int64(row) * fx.Int64(out_n) + fx.Int64(nbase + index))
            )

    @flyc.jit
    def launch(
        a: fx.Int64, w: fx.Int64, scales: fx.Int64, rows: fx.Int64, eids: fx.Int64, out: fx.Int64, stream: fx.Stream
    ):
        kernel(a, w, scales, rows, eids, out).launch(
            grid=(sorted_rows // 16 * nblocks, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def mxfp4_fp8_moe_gemm(a, packed_w, scales, row_ids, expert_ids, *, gate_up=False, out=None):
    """Run the experimental fused conversion + MoE GEMM on the current device.

    a is BF16 [rows,K]; row_ids and expert_ids are int32. See the builder for
    routing/weight layouts. The down projection returns sorted BF16 rows;
    routing weights and scatter/reduction belong to the caller.
    """
    tensors = (a, packed_w, scales, row_ids, expert_ids)
    if not a.is_cuda or any(t.device != a.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Inputs must be contiguous GPU tensors on the same device")
    if (a.dtype, packed_w.dtype, scales.dtype, row_ids.dtype, expert_ids.dtype) != (
        torch.bfloat16,
        torch.uint8,
        torch.uint8,
        torch.int32,
        torch.int32,
    ):
        raise TypeError("Expected BF16 A, uint8 weights/scales and int32 routing indices")
    if a.ndim != 2 or packed_w.ndim != 3 or row_ids.ndim != 1 or expert_ids.ndim != 1:
        raise ValueError("Expected A[rows,K], W[E,N,K/2], row_ids[S], expert_ids[S/16]")
    e, n, half_k = packed_w.shape
    k = a.shape[1]
    if half_k * 2 != k or tuple(scales.shape) != (e, n, k // 32) or expert_ids.numel() * 16 != row_ids.numel():
        raise ValueError("Weight/scale/routing dimensions do not match")
    shape = (row_ids.numel(), n // 2 if gate_up else n)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=a.device)
    if out.shape != shape or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be a contiguous BF16 tensor of the expected shape on the input device")
    if any(out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr() for t in tensors):
        raise ValueError("out must not alias an input")
    with torch.cuda.device(a.device):
        arch = get_rocm_arch().split(":")[0]
        launch = compile_mxfp4_fp8_moe_gemm(
            K=k, N=n, experts=e, a_rows=a.shape[0], sorted_rows=row_ids.numel(), gate_up=gate_up, arch=arch
        )
        _run_compiled(launch, *(t.data_ptr() for t in tensors), out.data_ptr(), torch.cuda.current_stream())
    return out
