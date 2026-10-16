# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Expand packed E2M1 weights to FP8 in registers, including on gfx942.

Each 32-bit word contains eight FP4 values, least-significant nibble first.
The helper returns an eight-element FP8 vector. MXFP4 E8M0 block scales
remain separate; this conversion does not apply them.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr
from kernels.common.tensor_shim import _to_raw as _raw
from kernels.moe.moe_2stage_a16wmix.utils import _fp4_perm


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
