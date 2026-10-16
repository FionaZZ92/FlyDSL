# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Weight-only FP4 -> FP8 correctness and MiMo-V2.6-Pro-RL benchmarks.

Run from the repository root on ROCm:
  python -m pytest tests/kernels/test_mxfp4_fp8_gfx942.py -m 'not benchmark'
  python -m pytest tests/kernels/test_mxfp4_fp8_gfx942.py -m benchmark -s

Benchmarks use random packed weights, without E8M0 scales or GEMM. Logical
gate_up is [E, 4096, 6144]; down is [E, 6144, 2048]. Packed input halves K.
E=8 is a contiguous group, E=48 assumes EP=8/TP=1, and E=384 is one layer.
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
from kernels.moe.mxfp4_fp8_gfx942 import fp4_to_fp8x8
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


def _expected_bytes(words, *, fnuz):
    # Decode numerically, independently of the kernel's byte-permute tables.
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=words.device)
    codes = (words.to(torch.int64)[:, None] >> (torch.arange(8, device=words.device) * 4)) & 15
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    return table[codes].to(dtype).view(torch.uint8)


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


def _check_weight_samples(src, dst, *, fnuz, generator):
    words = src.numel() // 4
    ids = torch.cat(
        (
            torch.arange(1024, device="cuda"),
            torch.arange(words - 1024, words, device="cuda"),
            torch.randint(words, (65536,), device="cuda", generator=generator),
        )
    )
    expected = _expected_bytes(src.view(torch.int32).flatten()[ids], fnuz=fnuz)
    torch.testing.assert_close(dst.view(-1, 8)[ids], expected, rtol=0, atol=0)


@pytest.mark.benchmark
@pytest.mark.large_shape
@pytest.mark.parametrize("experts", [1, 8, 48, 384])
@pytest.mark.parametrize("projection,n,k", [("gate_up", 4096, 6144), ("down", 6144, 2048)])
def test_mimo_weight_conversion_benchmark(experts, projection, n, k):
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":")[0]
    fnuz = arch == "gfx942"
    count = experts * n * k
    # Leave room for reference samples, runtime workspaces and the flush buffer.
    if torch.cuda.mem_get_info()[0] < count * 3 // 2 + 1024**3:
        pytest.skip("Insufficient free GPU memory for this weight shape")
    generator = torch.Generator(device="cuda").manual_seed(20261016)
    src = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=generator)
    dst = torch.empty((experts, n, k), dtype=torch.uint8, device="cuda")
    flush = torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda")
    launch = _make_launch(count // 8, fnuz=fnuz)

    def run():
        launch(src.data_ptr(), dst.data_ptr(), torch.cuda.current_stream())

    for _ in range(5):
        run()
    torch.cuda.synchronize()
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(30):
            run()
    dst.fill_(255)
    graph.replay()
    torch.cuda.synchronize()
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator)

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
    _check_weight_samples(src, dst, fnuz=fnuz, generator=generator)

    hot_us, cold_us = statistics.median(hot), statistics.median(cold)
    cold_tbps = (count * 3 // 2) / cold_us / 1e6
    print(
        f"\n{arch} E={experts} {projection} logical=[{experts},{n},{k}] "
        f"repeated_us={hot_us:.3f} flushed_us={cold_us:.3f} flushed_TBps={cold_tbps:.3f}"
    )
