# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Compile both projections to gfx942 without loading or executing GPU code.

Run from the repository root:
  COMPILE_ONLY=1 ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942 \
    python -m tests.kernels.compile_mxfp4_fp8_gfx942
Add FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/mxfp4-fp8-isa for target assembly.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.utils import env
from kernels.moe.mxfp4_fp8_gfx942 import compile_mxfp4_fp8_moe_gemm


def main():
    if not env.compile.compile_only or env.compile.arch != "gfx942":
        raise RuntimeError("Set COMPILE_ONLY=1 ARCH=gfx942 FLYDSL_GPU_ARCH=gfx942")
    for gate_up in (False, True):
        launch = compile_mxfp4_fp8_moe_gemm(
            K=128, N=64, experts=3, a_rows=37, sorted_rows=64, gate_up=gate_up, arch="gfx942"
        )
        compiled = flyc.compile(launch, 0, 0, 0, 0, 0, 0, fx.Stream(None))
        assert compiled is None, "compile-only mode must not create an executable callable"
        print(f"gfx942 {'gate/up + SiLU' if gate_up else 'down'}: compilation passed; no execution")


if __name__ == "__main__":
    main()
