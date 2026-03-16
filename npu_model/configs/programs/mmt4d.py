
from typing import List, Tuple

import math

import torch

from ...software import Instruction, Program


# ============================================================================
# Hardware tile dimensions (from NPU spec / DefaultHardwareConfig)
# ============================================================================
M0 = 64   # MRF depth: rows per activation tile
K0 = 32   # FP8 columns per activation tile / rows per weight tile
N0 = 16   # Columns per weight tile / output tile

# Tile sizes in bytes
A_TILE_BYTES = M0 * K0 * 1    # 64 × 32 FP8 = 2048 bytes
B_TILE_BYTES = K0 * N0 * 1    # 32 × 16 FP8 =  512 bytes
C_TILE_BYTES = M0 * N0 * 2    # 64 × 16 BF16 = 2048 bytes

# MRF register assignments
ACC_REG = 0    # accumulator for output tile
ACT_REG = 1    # activation (input) tile
WB_SLOT = 0    # weight buffer slot


def make_tiled_matmul_program(
    M: int,
    N: int,
    K: int,
    a_base: int = 0x0000,
    b_base: int | None = None,
    c_base: int | None = None,
) -> Tuple[List[Instruction], List[Tuple[int, torch.Tensor]], int]:
    """
    Generate the instruction sequence and memory regions for a tiled matmul.

    C[M, N] = A[M, K] @ B[K, N]

    Args:
        M, N, K: Matrix dimensions. Must be multiples of M0, N0, K0 respectively.
        a_base:  Memory base address for A tiles.
        b_base:  Memory base address for B tiles (auto-computed if None).
        c_base:  Memory base address for C tiles (auto-computed if None).

    Returns:
        (instructions, memory_regions, c_base) tuple.
    """
    # ---- Validate dimensions are tile-aligned ----
    assert M % M0 == 0, f"M={M} must be a multiple of M0={M0}"
    assert N % N0 == 0, f"N={N} must be a multiple of N0={N0}"
    assert K % K0 == 0, f"K={K} must be a multiple of K0={K0}"

    m_tiles = M // M0
    n_tiles = N // N0
    k_tiles = K // K0

    # ---- Compute memory layout ----
    total_a_bytes = m_tiles * k_tiles * A_TILE_BYTES
    total_b_bytes = k_tiles * n_tiles * B_TILE_BYTES

    if b_base is None:
        # Place B right after A, aligned to 4096
        b_base = ((a_base + total_a_bytes + 0xFFF) // 0x1000) * 0x1000
    if c_base is None:
        c_base = ((b_base + total_b_bytes + 0xFFF) // 0x1000) * 0x1000

    # ---- Helpers for tile address computation ----
    def a_tile_addr(m_idx: int, k_idx: int) -> int:
        """Address of A tile [m_idx, k_idx] in the mmt4d layout."""
        linear_idx = m_idx * k_tiles + k_idx
        return a_base + linear_idx * A_TILE_BYTES

    def b_tile_addr(k_idx: int, n_idx: int) -> int:
        """Address of B tile [k_idx, n_idx] in the mmt4d layout."""
        linear_idx = k_idx * n_tiles + n_idx
        return b_base + linear_idx * B_TILE_BYTES

    def c_tile_addr(m_idx: int, n_idx: int) -> int:
        """Address of C tile [m_idx, n_idx] in the mmt4d layout."""
        linear_idx = m_idx * n_tiles + n_idx
        return c_base + linear_idx * C_TILE_BYTES

    # ---- Generate instruction sequence ----
    instructions: List[Instruction] = []

    for m in range(m_tiles):
        for n in range(n_tiles):
            # Clear the accumulator register by self-subtracting.
            # After this, MRF[ACC_REG] is all zeros (BF16).
            instructions.append(
                Instruction(mnemonic="vsub", args={"vrd": ACC_REG, "vs1": ACC_REG, "vs2": ACC_REG})
            )

            for k in range(k_tiles):
                # --- Load weight tile B[k, n] into weight buffer ---
                instructions.append(
                    Instruction(
                        mnemonic="dma.load.mxu0",
                        args={
                            "rd": WB_SLOT,
                            "base": b_tile_addr(k, n),
                            "size": B_TILE_BYTES,
                            "flag": 0,
                        },
                    )
                )

                # --- Load activation tile A[m, k] into MRF ---
                instructions.append(
                    Instruction(
                        mnemonic="dma.load",
                        args={
                            "rd": ACT_REG,
                            "base": a_tile_addr(m, k),
                            "size": A_TILE_BYTES,
                            "flag": 1,
                        },
                    )
                )

                # Wait for both DMA loads to complete
                instructions.append(Instruction(mnemonic="dma.wait", args={"flag": 0}))
                instructions.append(Instruction(mnemonic="dma.wait", args={"flag": 1}))

                # --- Matrix multiply-accumulate ---
                # ACC_REG[64,16] += ACT_REG[64,32] @ WB[0][32,16]
                # This is the core mmt4d tile operation.
                instructions.append(
                    Instruction(
                        mnemonic="matmul.mxu0",
                        args={"rd": ACC_REG, "rs1": ACT_REG, "rs2": WB_SLOT},
                    )
                )

            # --- Store completed output tile C[m, n] ---
            instructions.append(
                Instruction(
                    mnemonic="dma.store",
                    args={
                        "rs1": ACC_REG,
                        "base": c_tile_addr(m, n),
                        "size": C_TILE_BYTES,
                        "flag": 0,
                    },
                )
            )
            instructions.append(Instruction(mnemonic="dma.wait", args={"flag": 0}))

    # Terminate
    instructions.append(Instruction(mnemonic="delay", args={"imm": 0}))

    # ---- Generate test data in mmt4d tiled layout ----
    # Create random FP8 test matrices
    A_flat = torch.randn(M, K).to(torch.float8_e4m3fn)
    B_flat = torch.randn(K, N).to(torch.float8_e4m3fn)

    # Rearrange A into tiled layout: [M/M0, K/K0, M0, K0] stored contiguously
    A_tiled_bytes = bytearray()
    for mi in range(m_tiles):
        for ki in range(k_tiles):
            tile = A_flat[mi * M0 : (mi + 1) * M0, ki * K0 : (ki + 1) * K0]
            A_tiled_bytes.extend(tile.contiguous().view(torch.uint8).numpy().tobytes())
    A_tiled = torch.frombuffer(bytearray(A_tiled_bytes), dtype=torch.uint8).clone()

    # Rearrange B into tiled layout: [K/K0, N/N0, K0, N0] stored contiguously
    B_tiled_bytes = bytearray()
    for ki in range(k_tiles):
        for ni in range(n_tiles):
            tile = B_flat[ki * K0 : (ki + 1) * K0, ni * N0 : (ni + 1) * N0]
            B_tiled_bytes.extend(tile.contiguous().view(torch.uint8).numpy().tobytes())
    B_tiled = torch.frombuffer(bytearray(B_tiled_bytes), dtype=torch.uint8).clone()

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (a_base, A_tiled.view(torch.float8_e4m3fn)),
        (b_base, B_tiled.view(torch.float8_e4m3fn)),
    ]

    # ---- Compute golden result ----
    golden_C = compute_golden_tiled_matmul(M, N, K, A_flat, B_flat)

    # Re-tile C into mmt4d layout for memory comparison
    C_tiled_bytes = bytearray()
    for mi in range(m_tiles):
        for ni in range(n_tiles):
            tile = golden_C[mi * M0 : (mi + 1) * M0, ni * N0 : (ni + 1) * N0]
            C_tiled_bytes.extend(tile.contiguous().view(torch.uint8).numpy().tobytes())
    C_tiled = torch.frombuffer(bytearray(C_tiled_bytes), dtype=torch.uint8).clone()
    golden_result = (c_base, C_tiled.view(torch.bfloat16))

    return instructions, memory_regions, c_base, golden_result


def compute_golden_tiled_matmul(
    M: int, N: int, K: int, A_flat: torch.Tensor, B_flat: torch.Tensor
) -> torch.Tensor:
    """
    Compute golden reference matching the hardware's tile-by-tile accumulation.

    Simulates exactly what the hardware does:
    - Reads activation MRF as FP8 (64×32)
    - Reads weight WB as FP8 (32×16)
    - Multiplies in FP16
    - Accumulates into BF16

    This ensures the golden result accounts for quantization artifacts
    from the FP8→FP16→BF16 pipeline.
    """
    m_tiles = M // M0
    n_tiles = N // N0
    k_tiles = K // K0

    C = torch.zeros(M, N, dtype=torch.bfloat16)

    for mi in range(m_tiles):
        for ni in range(n_tiles):
            acc = torch.zeros(M0, N0, dtype=torch.bfloat16)

            for ki in range(k_tiles):
                a_tile_fp8 = A_flat[
                    mi * M0 : (mi + 1) * M0, ki * K0 : (ki + 1) * K0
                ].to(torch.float8_e4m3fn)

                b_tile_fp8 = B_flat[
                    ki * K0 : (ki + 1) * K0, ni * N0 : (ni + 1) * N0
                ].to(torch.float8_e4m3fn)

                # Hardware pipeline: FP8 → FP16 multiply → FP16 add → BF16 writeback
                product_fp16 = a_tile_fp8.to(torch.float16) @ b_tile_fp8.to(torch.float16)
                acc_fp16 = acc.to(torch.float16) + product_fp16
                acc = acc_fp16.to(torch.bfloat16)

            C[mi * M0 : (mi + 1) * M0, ni * N0 : (ni + 1) * N0] = acc

    return C


# ============================================================================
# Default example: 128×32 = 128×64 @ 64×32
# Two tiles along M, two tiles along K, two tiles along N
# ============================================================================

EXAMPLE_M = 128
EXAMPLE_N = 32
EXAMPLE_K = 64

_instructions, _memory_regions, _c_base, _golden_result = make_tiled_matmul_program(
    M=EXAMPLE_M, N=EXAMPLE_N, K=EXAMPLE_K
)


class TiledMatmulProgram(Program):
    """
    Tiled matmul (mmt4d style): C[128,32] = A[128,64] @ B[64,32]

    This decomposes into:
        m_tiles=2, n_tiles=2, k_tiles=2
    for a total of 2×2×2 = 8 matmul instructions,
    with 2×2 = 4 output tiles stored.

    The data in memory is arranged in mmt4d (4D tiled) layout.
    """

    instructions: List[Instruction] = _instructions
    memory_regions: List[Tuple[int, torch.Tensor]] = _memory_regions
    golden_result: Tuple[int, torch.Tensor] = _golden_result


# ============================================================================
# Convenience: larger case studies matching real workload dimensions
# ============================================================================


def make_program_for_workload(name: str) -> Tuple[List[Instruction], List[Tuple[int, torch.Tensor]], int]:
    """
    Generate tiled matmul programs for workloads from the speed-of-light model.

    These dimensions come from actual Gemma/SigLIP model layers,
    rounded up to tile boundaries.
    """
    # Round up to nearest tile boundary
    def align(val: int, tile: int) -> int:
        return int(math.ceil(val / tile)) * tile

    workloads = {
        # name: (M, N, K) — original dimensions from the throughput model
        "gemma_mlp_up":         (align(816, M0), align(16384, N0), align(2048, K0)),
        "gemma_mlp_down":       (align(816, M0), align(2048, N0),  align(16384, K0)),
        "gemma_attn_q":         (align(816, M0), align(2048, N0),  align(2048, K0)),
        "siglip_attn_self":     (align(256, M0), align(1152, N0),  align(1152, K0)),
        "gemma_attn_kxq":       (align(816, M0), align(256, N0),   align(816, K0)),
        # Small / test cases
        "small_test":           (M0, N0, K0),           # 1 tile, no looping
        "medium_test":          (2 * M0, 2 * N0, 2 * K0),  # 2×2×2 = 8 tiles
    }

    if name not in workloads:
        raise ValueError(f"Unknown workload '{name}'. Available: {list(workloads.keys())}")

    M, N, K = workloads[name]
    return make_tiled_matmul_program(M, N, K)