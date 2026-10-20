# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Per-output-channel MXFP4 conversion: ranges, rounding, reconstruction and replay."""

import pytest
import torch

from kernels.moe.mxfp4_fp8_per_oc_gfx942 import compile_mxfp4_to_fp8_per_oc, mxfp4_to_fp8_per_oc

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


def _reference(packed, scales, dtype):
    # Decode in FP64: legal MXFP4 values can exceed FP32 before normalization.
    packed, scales = packed.cpu(), scales.cpu()
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    factors = torch.exp2(scales.double() - 127)
    finite = scales != 255
    bound = 6 * torch.where(finite, factors, 0).amax(dim=-1)
    oc_scale = torch.exp2(torch.ceil(torch.log2(bound / torch.finfo(dtype).max)))
    oc_scale = oc_scale.clamp_min(torch.finfo(torch.float32).tiny).float()
    factors = torch.where(finite, factors, float("nan"))
    weights = table[codes] * factors.repeat_interleave(32, dim=-1)
    normalized = (weights / oc_scale.double()[..., None]).float()
    return normalized.to(dtype), oc_scale, weights


@pytest.mark.parametrize("fnuz", [True, False], ids=["fnuz", "fn"])
def test_per_oc_all_finite_scale_pairs(fnuz):
    """Every finite channel max, every block code <= max, all 16 FP4 codes."""
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    # 257 groups exercises repeated wave scans and a one-lane final chunk.
    maxima = torch.arange(255)[:, None]
    codes = torch.minimum(torch.arange(257).clamp_max(254)[None, :], maxima)
    scales = codes.to(torch.uint8)[None].cuda()
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8)
    packed = packed.repeat(1, 255, 257 * 2).cuda()
    out = torch.empty((1, 255, 257 * 32), dtype=dtype, device="cuda")
    out_scale = torch.empty((1, 255), dtype=torch.float32, device="cuda")
    launch = compile_mxfp4_to_fp8_per_oc(channels=255, groups_per_channel=257, fnuz=fnuz)
    launch(packed.data_ptr(), scales.data_ptr(), out.data_ptr(), out_scale.data_ptr(), torch.cuda.current_stream())
    expected, expected_scale, _ = _reference(packed, scales, dtype)
    torch.testing.assert_close(out.cpu().view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(out_scale.cpu(), expected_scale, rtol=0, atol=0)
    assert torch.isfinite(out.float()).all()
    assert out.float().abs().max() <= (192 if fnuz else 384)
    assert torch.isfinite(out_scale).all() and (out_scale >= torch.finfo(torch.float32).tiny).all()


@pytest.mark.parametrize("fnuz", [True, False], ids=["fnuz", "fn"])
def test_per_oc_nan_scales(fnuz):
    dtype = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    scales = torch.tensor([[[255, 0, 254], [255, 255, 255], [127, 255, 133]]], dtype=torch.uint8, device="cuda")
    packed = torch.arange(256, dtype=torch.int32).to(torch.uint8).repeat(2)[:144].reshape(1, 3, 48).cuda()
    out = torch.empty((1, 3, 96), dtype=dtype, device="cuda")
    out_scale = torch.empty((1, 3), dtype=torch.float32, device="cuda")
    launch = compile_mxfp4_to_fp8_per_oc(channels=3, groups_per_channel=3, fnuz=fnuz)
    launch(packed.data_ptr(), scales.data_ptr(), out.data_ptr(), out_scale.data_ptr(), torch.cuda.current_stream())
    expected, expected_scale, _ = _reference(packed, scales, dtype)
    # NaN payload/sign from the FP64 reference is not part of this API.
    actual, expected = out.cpu().float(), expected.float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(out_scale.cpu(), expected_scale, rtol=0, atol=0)
    assert torch.equal(torch.isnan(actual), (scales.cpu() == 255).repeat_interleave(32, dim=-1))


@pytest.mark.parametrize("shape", [(1, 1, 32), (2, 7, 96), (3, 17, 2080), (1, 3, 6144)])
def test_per_oc_reconstruction_and_graph(shape):
    e, n, k = shape
    generator = torch.Generator().manual_seed(719)
    packed = torch.randint(0, 256, (e, n, k // 2), generator=generator, dtype=torch.uint8).cuda()
    # Modest scale spread makes every original FP4 value exactly recoverable.
    scales = torch.randint(125, 136, (e, n, k // 32), generator=generator, dtype=torch.uint8).cuda()
    result, result_scale = mxfp4_to_fp8_per_oc(packed, scales)
    expected, expected_scale, weights = _reference(packed, scales, result.dtype)
    torch.testing.assert_close(result.cpu().view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(result_scale.cpu(), expected_scale, rtol=0, atol=0)
    reconstructed = result.cpu().double() * result_scale.cpu().double()[..., None]
    torch.testing.assert_close(reconstructed, weights, rtol=0, atol=0)
    # A per-channel scale can be applied after the dot product, before bias.
    activation = torch.randn((e, 2, k), generator=generator, dtype=torch.float64)
    actual_gemm = (activation @ result.cpu().double().transpose(-1, -2)) * result_scale.cpu().double()[:, None, :]
    torch.testing.assert_close(actual_gemm, activation @ weights.transpose(-1, -2), rtol=1e-12, atol=1e-10)

    storage = torch.full((result.numel() + 8,), 85, dtype=torch.uint8, device="cuda")
    scale_storage = torch.full((e * n + 2,), -17.0, dtype=torch.float32, device="cuda")
    out = storage[4:-4].view(result.dtype).reshape(shape)
    out_scale = scale_storage[1:-1].reshape(e, n)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q, s = mxfp4_to_fp8_per_oc(packed, scales, out=out, out_scale=out_scale)
    assert q is out and s is out_scale
    # Replay must recompute the channel max from the current input.
    scales.add_(1)
    graph.replay()
    expected, expected_scale, _ = _reference(packed, scales, result.dtype)
    torch.testing.assert_close(out.cpu().view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(out_scale.cpu(), expected_scale, rtol=0, atol=0)
    assert torch.all(storage[:4] == 85) and torch.all(storage[-4:] == 85)
    assert scale_storage[0] == -17 and scale_storage[-1] == -17


def test_per_oc_invalid_inputs():
    packed = torch.zeros((1, 2, 32), dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 2, 2), 133, dtype=torch.uint8, device="cuda")
    out, out_scale = mxfp4_to_fp8_per_oc(packed, scales)
    with pytest.raises(TypeError, match="uint8"):
        mxfp4_to_fp8_per_oc(packed.float(), scales)
    with pytest.raises(ValueError, match="same ROCm"):
        mxfp4_to_fp8_per_oc(packed, scales.cpu())
    with pytest.raises(ValueError, match="scales"):
        mxfp4_to_fp8_per_oc(packed, scales[..., :1].contiguous())
    with pytest.raises(ValueError, match="contiguous"):
        mxfp4_to_fp8_per_oc(packed.transpose(1, 2), scales)
    with pytest.raises(ValueError, match="K divisible"):
        mxfp4_to_fp8_per_oc(packed[..., :17].contiguous(), scales)
    with pytest.raises(ValueError, match="out must"):
        mxfp4_to_fp8_per_oc(packed, scales, out=out.float())
    with pytest.raises(ValueError, match="out_scale must"):
        mxfp4_to_fp8_per_oc(packed, scales, out_scale=out_scale.half())
    with pytest.raises(ValueError, match="alias"):
        mxfp4_to_fp8_per_oc(
            packed, scales, out=out, out_scale=out.view(torch.uint8).flatten()[:8].view(torch.float32).reshape(1, 2)
        )
    backing = torch.empty(256, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="alias"):
        mxfp4_to_fp8_per_oc(
            backing[:64].reshape_as(packed), scales, out=backing[64:192].view(out.dtype).reshape_as(out)
        )
    with pytest.raises(ValueError, match="aligned"):
        mxfp4_to_fp8_per_oc(backing[1:65].reshape_as(packed), scales)
