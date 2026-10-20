# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compare scaled MXFP4 conversion with per-OC normalization on MiMo shapes.

From the repository root:
  HIP_VISIBLE_DEVICES=0 python -m tests.kernels.bench_mxfp4_fp8_per_oc --output results.json

GPU-event timings include the complete conversion kernel, including per-OC
reduction and scale stores. Allocation, compilation, validation and the 512 MiB
cache flush are excluded. Methods alternate order between rounds. Inputs use
synthetic E8M0 codes 123..131, so the baseline does not overflow.
"""

import argparse
import datetime
import hashlib
import json
import statistics
from pathlib import Path

import torch

from kernels.moe.mxfp4_fp8_gfx942 import compile_mxfp4_to_fp8
from kernels.moe.mxfp4_fp8_per_oc_gfx942 import compile_mxfp4_to_fp8_per_oc


def benchmark(experts, n, k, *, fnuz, repeats, hot_rounds, cold_rounds):
    count = experts * n * k
    generator = torch.Generator(device="cuda").manual_seed(20261019)
    src = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device="cuda", generator=generator)
    scales = torch.randint(123, 132, (experts, n, k // 32), dtype=torch.uint8, device="cuda", generator=generator)
    dst = torch.empty((experts, n, k), dtype=torch.uint8, device="cuda")
    oc_scale = torch.empty((experts, n), dtype=torch.float32, device="cuda")
    flush = torch.zeros(512 * 1024**2, dtype=torch.uint8, device="cuda")
    baseline = compile_mxfp4_to_fp8(groups=count // 32, fnuz=fnuz)
    per_oc = compile_mxfp4_to_fp8_per_oc(channels=experts * n, groups_per_channel=k // 32, fnuz=fnuz)

    def run_baseline():
        baseline(src.data_ptr(), scales.data_ptr(), dst.data_ptr(), torch.cuda.current_stream())

    def run_per_oc():
        per_oc(src.data_ptr(), scales.data_ptr(), dst.data_ptr(), oc_scale.data_ptr(), torch.cuda.current_stream())

    methods = {"baseline": run_baseline, "per_oc": run_per_oc}
    words = src.numel() // 4
    ids = torch.cat(
        (
            torch.arange(1024, device="cuda"),
            torch.arange(words - 1024, words, device="cuda"),
            torch.randint(words, (65536,), device="cuda", generator=generator),
        )
    )
    # Numerically decode independent reference samples; compare FP8 bytes.
    sampled_words = src.view(torch.int32).flatten()[ids].cpu().long()
    codes = (sampled_words[:, None] >> (4 * torch.arange(8))) & 15
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
    values = table[codes] * torch.exp2(scales.flatten()[ids // 4].cpu().float()[:, None] - 127)
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    bound = 6 * torch.exp2(scales.amax(dim=-1).float() - 127)
    expected_scale = torch.exp2(torch.ceil(torch.log2(bound / torch.finfo(dtype).max)))
    normalized = values / expected_scale.flatten()[ids // (k // 8)].cpu()[:, None]
    expected = {
        "baseline": values.to(dtype).view(torch.uint8),
        "per_oc": normalized.to(dtype).view(torch.uint8),
    }

    def validate(name):
        torch.testing.assert_close(dst.view(-1, 8)[ids].cpu(), expected[name], rtol=0, atol=0)
        if name == "per_oc":
            torch.testing.assert_close(oc_scale, expected_scale, rtol=0, atol=0)

    graphs, singles = {}, {}
    for name, run in methods.items():
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        validate(name)
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            for _ in range(repeats):
                run()
        singles[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(singles[name]):
            run()
        # Check that capture/replay really writes both output tensors.
        dst.fill_(255)
        oc_scale.fill_(-1)
        graphs[name].replay()
        validate(name)

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {name: {"repeated_us": [], "flushed_us": []} for name in methods}
    for mode, rounds, graph_set, divisor in (
        ("repeated_us", hot_rounds, graphs, repeats),
        ("flushed_us", cold_rounds, singles, 1),
    ):
        for r in range(rounds):
            order = list(methods) if r % 2 == 0 else list(reversed(methods))
            for name in order:
                if mode == "flushed_us":
                    flush.add_(1)
                start.record()
                graph_set[name].replay()
                end.record()
                end.synchronize()
                samples[name][mode].append(start.elapsed_time(end) * 1000 / divisor)
    for name in methods:
        singles[name].replay()
        validate(name)
    result = {"logical_shape": [experts, n, k], "samples_us": samples, "sampled_words": ids.numel()}
    for name in methods:
        result[name] = {mode: statistics.median(times) for mode, times in samples[name].items()}
    result["overhead_percent"] = {
        mode: 100 * (result["per_oc"][mode] / result["baseline"][mode] - 1) for mode in ("repeated_us", "flushed_us")
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, nargs="+", default=[1, 8, 48, 62, 384])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--hot-rounds", type=int, default=7)
    parser.add_argument("--cold-rounds", type=int, default=28)
    args = parser.parse_args()
    if min(args.experts + [args.repeats, args.hot_rounds, args.cold_rounds]) <= 0:
        parser.error("experts and iteration counts must be positive")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    arch = props.gcnArchName.split(":")[0]
    if arch not in ("gfx942", "gfx950"):
        parser.error("Requires gfx942 or gfx950")
    root = Path(__file__).resolve().parents[2]
    sources = (
        "kernels/moe/mxfp4_fp8_gfx942.py",
        "kernels/moe/mxfp4_fp8_per_oc_gfx942.py",
        "tests/kernels/bench_mxfp4_fp8_per_oc.py",
    )
    # parents[2] is tests' parent, the repository root.
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": props.name,
        "arch": arch,
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "methodology": {
            "seed": 20261019,
            "e8m0_codes_inclusive": [123, 131],
            "synthetic_inputs": True,
            "hot_rounds": args.hot_rounds,
            "replays_per_hot_graph": args.repeats,
            "cold_rounds": args.cold_rounds,
            "flush_bytes": 512 * 1024**2,
            "alternating_method_order": True,
            "timing": "HIP graph + GPU events, medians, complete kernel only",
            "clocks_locked": False,
            "tp": 1,
        },
        "results": [],
    }
    print(json.dumps({k: report[k] for k in ("device", "arch", "torch", "methodology")}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for experts in args.experts:
        for projection, n, k in (("gate_up", 4096, 6144), ("down", 6144, 2048)):
            row = benchmark(
                experts,
                n,
                k,
                fnuz=arch == "gfx942",
                repeats=args.repeats,
                hot_rounds=args.hot_rounds,
                cold_rounds=args.cold_rounds,
            )
            row["projection"] = projection
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if key != "samples_us"}), flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
