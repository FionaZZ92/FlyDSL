# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Weight-only FP4 -> FP8 correctness and MiMo-V2.6-Pro-RL benchmarks.

Run from the repository root on ROCm:
  python -m pytest tests/kernels/test_mxfp4_fp8_gfx942.py -m 'not benchmark'
  python -m pytest tests/kernels/test_mxfp4_fp8_gfx942.py -m benchmark -s

Benchmarks use random packed weights, with optional E8M0 scales, without GEMM. Logical
gate_up is [E, 4096, 6144]; down is [E, 6144, 2048]. Packed input halves K.
E=8 is a contiguous group, E=48 assumes EP=8/TP=1, and E=384 is one layer.
E=62 is a contiguous group of complete experts (TP=1). Scaled benchmarks
use synthetic E8M0 codes 123..131; correctness covers all 256 encodings.
The largest case needs about 14 GiB of free GPU memory.
"""

import functools
import statistics

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.moe.mxfp4_fp8_gfx942 import compile_mxfp4_to_fp8, fp4_to_fp8x8, mxfp4_to_fp8
from kernels.moe.mxfp_moe.mxfp4_gemm_common import global_typed_ptr

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="Requires a ROCm GPU"),
]


@pytest.fixture(autouse=True)
def _require_gfx942_or_gfx950():
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("Requires gfx942 or gfx950")


@functools.cache
def _make_launch(words, *, fnuz=True):
    threads, unroll = 256, 4
    assert words > 0 and words % (threads * unroll) == 0

    @flyc.kernel
    def convert(src: fx.Int64, dst: fx.Int64):
        # 64-bit addressing also covers the full 384-expert gate/up tensor.
        base = fx.Int64(gpu.block_id("x")) * fx.Int64(threads * unroll)
        lane = fx.Int64(gpu.thread_id("x"))
        sp = global_typed_ptr(src, T.i32)
        dp = global_typed_ptr(dst, T.i32)
        for u in range_constexpr(unroll):
            i = base + lane + fx.Int64(u * threads)
            result = fp4_to_fp8x8(fx.Int32(sp[i]), fnuz=fnuz).bitcast(fx.Int32)
            for j in range_constexpr(2):
                fx.ptr_store(result[j], fx.add_offset(dp, i * fx.Int64(2) + fx.Int64(j)))

    @flyc.jit
    def launch(src: fx.Int64, dst: fx.Int64, stream: fx.Stream):
        convert(src, dst).launch(grid=(words // (threads * unroll), 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def _expected_bytes(words, *, fnuz, scales=None):
    # Decode numerically, independently of the kernel's byte-permute tables.
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=words.device)
    codes = (words.to(torch.int64)[:, None] >> (torch.arange(8, device=words.device) * 4)) & 15
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    values = table[codes]
    if scales is not None:
        factors = torch.exp2(scales.float() - 127)
        factors = torch.where(scales == 255, float("nan"), factors)
        values = values * factors[:, None]
    return values.to(dtype).view(torch.uint8)


@pytest.mark.parametrize("fnuz", [True, False], ids=["fnuz", "fn"])
@pytest.mark.parametrize("pattern", ["all_codes", "random"])
def test_fp4_to_fp8x8(fnuz, pattern):
    if pattern == "all_codes":
        # Every possible byte at all four positions, including both zero signs.
        byte = torch.arange(256, dtype=torch.int64)
        words = torch.cat([(byte << (8 * p)) | (0x87654321 & ~(255 << (8 * p))) for p in range(4)])
        words = words.to(torch.int32).cuda()
    else:
        generator = torch.Generator(device="cuda").manual_seed(731)
        words = torch.randint(-(2**31), 2**31, (4096,), dtype=torch.int32, device="cuda", generator=generator)
    out = torch.empty((words.numel(), 8), device="cuda", dtype=torch.uint8)
    launch = _make_launch(words.numel(), fnuz=fnuz)
    launch(words.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    expected = _expected_bytes(words.cpu(), fnuz=fnuz)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # Capture on PyTorch's capture stream and verify replay actually writes.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(words.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    out.fill_(255)
    graph.replay()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def _check_weight_samples(src, dst, *, fnuz, generator, scales=None):
    words = src.numel() // 4
    ids = torch.cat(
        (
            torch.arange(1024, device="cuda"),
            torch.arange(words - 1024, words, device="cuda"),
            torch.randint(words, (65536,), device="cuda", generator=generator),
        )
    )
    sampled_scales = scales.flatten()[ids // 4] if scales is not None else None
    expected = _expected_bytes(src.view(torch.int32).flatten()[ids], fnuz=fnuz, scales=sampled_scales)
    torch.testing.assert_close(dst.view(torch.uint8).view(-1, 8)[ids], expected, rtol=0, atol=0)


@pytest.mark.parametrize("fnuz", [True, False], ids=["fnuz", "fn"])
def test_scaled_all_codes(fnuz):
    """All E8M0 encodings crossed with every packed byte at every position."""
    byte = torch.arange(256, dtype=torch.int64)
    words = torch.cat([(byte << (8 * p)) | (0x87654321 & ~(255 << (8 * p))) for p in range(4)])
    words = words.to(torch.int32).repeat(256, 1)
    scales = torch.arange(256, dtype=torch.uint8)[:, None].expand(256, 256).contiguous()
    src, gpu_scales = words.cuda(), scales.cuda()
    out = torch.empty((src.numel(), 8), dtype=torch.uint8, device="cuda")
    launch = compile_mxfp4_to_fp8(groups=scales.numel(), fnuz=fnuz)
    launch(src.data_ptr(), gpu_scales.data_ptr(), out.data_ptr(), torch.cuda.current_stream())
    expected = _expected_bytes(words.flatten(), fnuz=fnuz, scales=scales.flatten().repeat_interleave(4))
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 1, 32), (2, 7, 96), (3, 17, 160), (3, 17, 192)])
def test_scaled_weight_conversion(shape):
    e, n, k = shape
    generator = torch.Generator().manual_seed(731)
    src = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, generator=generator).cuda()
    scales = torch.randint(115, 136, (e, n, k // 32), dtype=torch.uint8, generator=generator).cuda()
    actual = mxfp4_to_fp8(src, scales)
    fnuz = actual.dtype == torch.float8_e4m3fnuz
    expected = _expected_bytes(
        src.cpu().view(torch.int32).flatten(), fnuz=fnuz, scales=scales.cpu().flatten().repeat_interleave(4)
    )
    torch.testing.assert_close(actual.view(torch.uint8).cpu().view(-1, 8), expected, rtol=0, atol=0)
    # Guard words catch writes beyond a partial final thread block.
    storage = torch.full((actual.numel() + 8,), 85, dtype=torch.uint8, device="cuda")
    out = storage[4:-4].view(actual.dtype).reshape(shape)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mxfp4_to_fp8(src, scales, out=out)
    graph.replay()
    torch.testing.assert_close(out.view(torch.uint8).cpu().view(-1, 8), expected, rtol=0, atol=0)
    assert torch.all(storage[:4] == 85) and torch.all(storage[-4:] == 85)


def test_scaled_invalid_inputs():
    src = torch.zeros((1, 2, 32), dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 2, 2), 127, dtype=torch.uint8, device="cuda")
    with pytest.raises(TypeError, match="uint8"):
        mxfp4_to_fp8(src.float(), scales)
    with pytest.raises(ValueError, match="scales"):
        mxfp4_to_fp8(src, scales[:, :, :1].contiguous())
    with pytest.raises(ValueError, match="contiguous"):
        mxfp4_to_fp8(src.transpose(1, 2), scales)
    with pytest.raises(ValueError, match="K divisible"):
        mxfp4_to_fp8(torch.zeros((1, 1, 17), dtype=torch.uint8, device="cuda"), scales)
    with pytest.raises(ValueError, match="out must"):
        mxfp4_to_fp8(src, scales, out=torch.empty((1, 2, 64), dtype=torch.float32, device="cuda"))


@pytest.mark.benchmark
@pytest.mark.large_shape
@pytest.mark.parametrize("experts", [1, 8, 48, 62, 384])
@pytest.mark.parametrize("scaled", [False, True], ids=["unscaled", "scaled"])
@pytest.mark.parametrize("projection,n,k", [("gate_up", 4096, 6144), ("down", 6144, 2048)])
def test_mimo_weight_conversion_benchmark(experts, projection, n, k, scaled):
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    fnuz = arch == "gfx942"
    count = experts * n * k
    # Leave room for reference samples, runtime workspaces and the flush buffer.
    traffic = count * 3 // 2 + (count // 32 if scaled else 0)
    if torch.cuda.mem_get_info()[0] < traffic + 1024**3:
        pytest.skip("Insufficient free GPU memory for this weight shape")
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    src = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=generator)
    dst = torch.empty((experts, n, k), dtype=torch.uint8, device="cuda")
    scales = None
    if scaled:
        # Synthetic scales 2**[-4,4], with finite, exactly representable output.
        scales = torch.randint(123, 132, (experts, n, k // 32), dtype=torch.uint8, device="cuda", generator=generator)
    flush = torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda")
    launch = compile_mxfp4_to_fp8(groups=count // 32, fnuz=fnuz) if scaled else _make_launch(count // 8, fnuz=fnuz)

    def run():
        if scaled:
            launch(src.data_ptr(), scales.data_ptr(), dst.data_ptr(), torch.cuda.current_stream())
        else:
            launch(src.data_ptr(), dst.data_ptr(), torch.cuda.current_stream())

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator, scales=scales)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(30):
            run()
    dst.fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator, scales=scales)

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    hot = []
    for _ in range(7):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        hot.append(start.elapsed_time(end) * 1000 / 30)

    single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(single):
        run()
    cold = []
    for _ in range(28):
        flush.add_(1)
        start.record()
        single.replay()
        end.record()
        end.synchronize()
        cold.append(start.elapsed_time(end) * 1000)
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator, scales=scales)

    hot_us, cold_us = statistics.median(hot), statistics.median(cold)
    cold_tbps = traffic / cold_us / 1e6
    print(
        f"\n{arch} E={experts} {projection} scaled={scaled} logical=[{experts},{n},{k}] "
        f"repeated_us={hot_us:.3f} flushed_us={cold_us:.3f} flushed_TBps={cold_tbps:.3f}"
    )
