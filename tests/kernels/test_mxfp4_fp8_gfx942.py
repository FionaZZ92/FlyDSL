# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.moe.mxfp4_fp8_gfx942 import fp4_to_fp8x8, mxfp4_fp8_moe_gemm
from kernels.moe.mxfp_moe.mxfp4_gemm_common import global_typed_ptr

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU runtime tests require ROCm hardware")


def _dtype():
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("Requires gfx942 or gfx950")
    return torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn


def _decode(w, scales):
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=w.device)
    codes = torch.stack((w & 15, w >> 4), -1).flatten(-2).long()
    s = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
    return table[codes] * s


def _quantize(a):
    dtype = _dtype()
    blocks = a.float().reshape(a.shape[0], -1, 32)
    limit = 128 if dtype == torch.float8_e4m3fnuz else 256
    target = (blocks.abs().amax(-1, keepdim=True) / limit).clamp_min(2.0**-120)
    scale = torch.exp2(torch.ceil(torch.log2(target)))
    return ((blocks / scale).to(dtype).float() * scale).reshape(a.shape)


def _reference(a, w, scales, row_ids, eids, gate_up=False):
    aq, weights = _quantize(a), _decode(w, scales)
    n = w.shape[1]
    out = torch.zeros(row_ids.numel(), n // 2 if gate_up else n, device=a.device)
    for block, expert in enumerate(eids.cpu().tolist()):
        if not 0 <= expert < w.shape[0]:
            continue
        rows = row_ids[block * 16 : (block + 1) * 16].long()
        valid = (rows >= 0) & (rows < a.shape[0])
        value = aq[rows.clamp(0, a.shape[0] - 1)] @ weights[expert].T
        if gate_up:
            gate, up = value.chunk(2, -1)
            value = torch.nn.functional.silu(gate) * up
        out[block * 16 : (block + 1) * 16] = torch.where(valid[:, None], value, 0)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("fnuz", [True, False])
def test_all_fp4_codes_and_packing(fnuz):
    """Test every byte in every position, mixed signs and signed zero on GPU."""
    # Every possible pair of nibbles in each byte position; other bytes vary.
    words = torch.arange(256, dtype=torch.int64)
    words = torch.cat([(words << (8 * p)) | (0x87654321 & ~(255 << (8 * p))) for p in range(4)])
    words = words.to(torch.int32).cuda()
    out = torch.empty((1024, 8), device="cuda", dtype=torch.uint8)

    @flyc.kernel
    def convert(src: fx.Int64, dst: fx.Int64):
        i = fx.Int32(gpu.block_id("x")) * fx.Int32(64) + fx.Int32(gpu.thread_id("x"))
        word = fx.Int32(global_typed_ptr(src, T.i32)[i])
        result = fp4_to_fp8x8(word, fnuz=fnuz).bitcast(fx.Int32)
        ptr = global_typed_ptr(dst, T.i32)
        for j in range_constexpr(2):
            fx.ptr_store(result[j], fx.add_offset(ptr, i * fx.Int32(2) + fx.Int32(j)))

    @flyc.jit
    def launch(src: fx.Int64, dst: fx.Int64, stream: fx.Stream):
        convert(src, dst).launch(grid=(16, 1, 1), block=(64, 1, 1), stream=stream)

    launch(words.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    codes = (words.cpu().to(torch.int64)[:, None] >> (torch.arange(8) * 4)) & 15
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    expected = table[codes].to(dtype).view(torch.uint8)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("gate_up", [False, True])
@pytest.mark.parametrize("k,n", [(32, 32), (128, 64), (256, 128)])
def test_grouped_projection(k, n, gate_up):
    _dtype()
    torch.manual_seed(731)
    a = torch.randn(37, k, device="cuda", dtype=torch.bfloat16)
    a[0].zero_()
    a[1] *= 256
    a[2] *= 2.0**-20
    w = torch.randint(0, 256, (3, n, k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(120, 128, (3, n, k // 32), device="cuda", dtype=torch.uint8)
    rows = torch.randint(0, 37, (64,), device="cuda", dtype=torch.int32)
    rows[:3] = torch.arange(3, device="cuda", dtype=torch.int32)
    rows[10:16] = -1
    rows[35] = 37
    eids = torch.tensor([2, 0, 1, -1], device="cuda", dtype=torch.int32)
    actual = mxfp4_fp8_moe_gemm(a, w, scales, rows, eids, gate_up=gate_up)
    expected = _reference(a, w, scales, rows, eids, gate_up)
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.0625)
    assert torch.count_nonzero(actual[10:16]) == 0
    assert torch.count_nonzero(actual[48:]) == 0


def test_two_stage_moe_with_topk_and_padding():
    _dtype()
    torch.manual_seed(19)
    tokens, hidden, inter, experts, topk = 13, 64, 32, 3, 2
    a = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    routes = torch.stack((torch.arange(tokens) % experts, (torch.arange(tokens) + 1) % experts), -1)
    route_weights = torch.softmax(torch.randn(tokens, topk, device="cuda"), -1)
    ids, eids, slots = [], [], []
    for e in range(experts):
        selected = (routes == e).nonzero().tolist()
        pad = (-len(selected)) % 16
        ids.extend([t for t, _ in selected] + [-1] * pad)
        slots.extend([t * topk + slot for t, slot in selected] + [-1] * pad)
        eids.extend([e] * ((len(selected) + pad) // 16))
    rows = torch.tensor(ids, device="cuda", dtype=torch.int32)
    eid = torch.tensor(eids, device="cuda", dtype=torch.int32)
    w1 = torch.randint(0, 256, (experts, inter * 2, hidden // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, inter // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.randint(120, 125, (experts, inter * 2, hidden // 32), device="cuda", dtype=torch.uint8)
    s2 = torch.randint(120, 125, (experts, hidden, inter // 32), device="cuda", dtype=torch.uint8)
    mid = mxfp4_fp8_moe_gemm(a, w1, s1, rows, eid, gate_up=True)
    expected_mid = _reference(a, w1, s1, rows, eid, True)
    torch.testing.assert_close(mid, expected_mid, rtol=0.016, atol=0.015625)
    down_rows = torch.arange(rows.numel(), device="cuda", dtype=torch.int32)
    down_rows[rows < 0] = -1
    down = mxfp4_fp8_moe_gemm(mid, w2, s2, down_rows, eid)
    # Stage2 is compared using its actual BF16 input, isolating its quantization.
    expected_down = _reference(mid, w2, s2, down_rows, eid)
    torch.testing.assert_close(down, expected_down, rtol=0.016, atol=0.015625)
    slot_ids = torch.tensor(slots, device="cuda")
    valid = slot_ids >= 0
    flat_slot = slot_ids[valid]
    weighted = down[valid].float() * route_weights.flatten()[flat_slot, None]
    expected_weighted = expected_down[valid].float() * route_weights.flatten()[flat_slot, None]
    out = torch.zeros(tokens, hidden, device="cuda").index_add_(0, flat_slot // topk, weighted)
    ref = torch.zeros_like(out).index_add_(0, flat_slot // topk, expected_weighted)
    torch.testing.assert_close(out, ref, rtol=0.016, atol=0.015625)
