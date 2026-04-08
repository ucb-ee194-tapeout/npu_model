"""PyTorch reference implementations for SmolVLA kernel ops.

These functions compute the expected (golden) output for NPU kernel programs
derived from the SmolVLA model decomposition. Each function corresponds to
a linalg.generic pattern found in the compiled MLIR at global-optimization
level.

See: merlin/benchmarks/SaturnNPU/ for the full model decomposition.
"""

import torch


def silu_reference(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation: x * sigmoid(x).

    In SmolVLA's compiled MLIR (module.4.global-optimization.mlir),
    this appears as a linalg.generic with body:

        %neg  = arith.negf %x : bf16
        %exp  = math.exp %neg : bf16
        %one  = arith.constant 1.0 : bf16
        %denom = arith.addf %one, %exp : bf16
        %result = arith.divf %x, %denom : bf16

    Used in: Gemma MLP layers (32 instances in SmolVLA).
    """
    # Compute in float32 for reference accuracy, cast back to input dtype.
    return (x.float() * torch.sigmoid(x.float())).to(x.dtype)
