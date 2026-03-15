from typing import List, Tuple
from ...software import (
    Instruction,
    Program,
)
import torch
import math
from npu_model.workload.gemma_blocks import gemma_layer_generic_fp8


SEQ_LEN = 64
DIM_MODEL = 32
HIDDEN_STATES_DATA = torch.ones((SEQ_LEN, DIM_MODEL), dtype=torch.float8_e4m3fn)


# Weights for Attention
DIM_HEAD = 16
QUERY_WEIGHT_DATA = torch.ones((DIM_MODEL, DIM_HEAD), dtype=torch.float8_e4m3fn)
KEY_WEIGHT_DATA = torch.ones((DIM_MODEL, DIM_HEAD), dtype=torch.float8_e4m3fn)
VALUE_WEIGHT_DATA = torch.ones((DIM_MODEL, DIM_HEAD), dtype=torch.float8_e4m3fn)
OUT_PROJ_WEIGHT_DATA = torch.ones((DIM_HEAD, DIM_MODEL), dtype=torch.float8_e4m3fn)


# Weights for MLP
DIM_FF = 16
GATE_WEIGHT_DATA = torch.ones((DIM_MODEL, DIM_FF), dtype=torch.float8_e4m3fn)
UP_WEIGHT_DATA = torch.ones((DIM_MODEL, DIM_FF), dtype=torch.float8_e4m3fn)
DOWN_WEIGHT_DATA = torch.ones((DIM_FF, DIM_MODEL), dtype=torch.float8_e4m3fn)


# Constants
EPS_DATA = torch.full((SEQ_LEN, DIM_MODEL // 2), 1e-6, dtype=torch.bfloat16)
DIVISOR_DATA = torch.full((SEQ_LEN, DIM_MODEL // 2), 1.0 / (DIM_MODEL // 2), dtype=torch.bfloat16)
SCALING_DATA = torch.full((SEQ_LEN, DIM_HEAD), 1.0 / math.sqrt(float(DIM_HEAD)), dtype=torch.bfloat16)


# Memory layout - assumes fp8
HIDDEN_STATES_BASE = 0x0000 # SEQ_LEN * DIM_MODEL bytes
QUERY_WEIGHT_BASE = 0x0800 # DIM_MODEL * DIM_HEAD bytes
KEY_WEIGHT_BASE = 0x0A00 # DIM_MODEL * DIM_HEAD bytes
VALUE_WEIGHT_BASE = 0x0C00 # DIM_MODEL * DIM_HEAD bytes
OUT_PROJ_WEIGHT_BASE = 0x0E00 # DIM_HEAD * DIM_MODEL bytes
GATE_WEIGHT_BASE = 0x1000 # DIM_MODEL * DIM_FF bytes
UP_WEIGHT_BASE = 0x1200 # DIM_MODEL * DIM_FF bytes
DOWN_WEIGHT_BASE = 0x1400 # DIM_FF * DIM_MODEL bytes
OUTPUT_BASE = 0x1600 # SEQ_LEN * DIM_MODEL bytes

EPS_BASE = 0x1E00 # SEQ_LEN * (DIM_MODEL // 2) * 2 bytes
DIVISOR_BASE = 0x2600 # SEQ_LEN * (DIM_MODEL // 2) * 2 bytes
SCALING_BASE = 0x2E00 # SEQ_LEN * DIM_HEAD * 2 bytes
SCRATCH_BASE = 0x3600


class GemmaLayerProgram(Program):
    """
    Gemma complete layer program (generic layer - update as we go).
    Assumes the hidden states is in fp8 and have been scaled by some factor α.
    RMSNorm -> Single-head Attention -> Residual Connection -> RMSNorm -> GEGLU MLP -> Residual Connection
    """

    instructions: List[Instruction] = [
        # Loads the upper half of hidden states.
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": HIDDEN_STATES_BASE,
                "size": (HIDDEN_STATES_DATA.numel() // 2) * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        # Loads the lower half of hidden states.
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 1,
                "base": (HIDDEN_STATES_BASE + 0x0400),
                "size": (HIDDEN_STATES_DATA.numel() // 2) * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        # Loads the epsilon.
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 2,
                "base": EPS_BASE,
                "size": (HIDDEN_STATES_DATA.numel() // 2) * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        # Loads the divisor.
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 3,
                "base": DIVISOR_BASE,
                "size": (HIDDEN_STATES_DATA.numel() // 2) * torch.bfloat16.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        
        # Upcast the upper half for rmsnorm.
        # TODO: need the original scaling factor α.
        Instruction(mnemonic="vcast.up", args={"vrd": 4, "vs1": 0}),
        # Upcast the lower half for rmsnorm.
        Instruction(mnemonic="vcast.up", args={"vrd": 5, "vs1": 1}),
        
        # [rmsnorm] - Upper half
        # x_sq = x * x
        Instruction(mnemonic="vmul", args={"vrd": 16, "vs1": 4, "vs2": 4}),
        # Row-wise sum via vrot.reduce.sum -> full (16,64), then reduce + broadcast
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 17, "vs1": 16}),
        # var = sum_sq / row_size = sum_sq * (1/row_size)
        Instruction(mnemonic="vmul", args={"vrd": 19, "vs1": 17, "vs2": 3}),
        # var_eps = var + eps
        Instruction(mnemonic="vadd", args={"vrd": 20, "vs1": 19, "vs2": 2}),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt", args={"vrd": 21, "vs1": 20}),
        Instruction(mnemonic="vrcp", args={"vrd": 22, "vs1": 21}),
        # output = x * rsqrt
        Instruction(mnemonic="vmul", args={"vrd": 23, "vs1": 4, "vs2": 22}),
        
        # [rmsnorm] - Lower half
        # x_sq = x * x
        Instruction(mnemonic="vmul", args={"vrd": 16, "vs1": 5, "vs2": 5}),
        # Row-wise sum via vrot.reduce.sum -> full (16,64), then reduce + broadcast
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 17, "vs1": 16}),
        # var = sum_sq / row_size = sum_sq * (1/row_size)
        Instruction(mnemonic="vmul", args={"vrd": 19, "vs1": 17, "vs2": 3}),
        # var_eps = var + eps
        Instruction(mnemonic="vadd", args={"vrd": 20, "vs1": 19, "vs2": 2}),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt", args={"vrd": 21, "vs1": 20}),
        Instruction(mnemonic="vrcp", args={"vrd": 22, "vs1": 21}),
        # output = x * rsqrt
        Instruction(mnemonic="vmul", args={"vrd": 24, "vs1": 5, "vs2": 22}),
        
        # [rmsnorm] - quantize back to fp8, then store in SCRATCH_BASE to load them all at once for the later matmuls.
        # TODO: will need to compute a new scaling factor since rmsnorm is nonlinear.
        Instruction(mnemonic="vcast.down", args={"vrd": 6, "vs1": 23}),
        Instruction(mnemonic="vcast.down", args={"vrd": 7, "vs1": 24}),
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 6,
                "base": SCRATCH_BASE,
                "size": 64 * 32 * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 7,
                "base": SCRATCH_BASE + 0x0800,
                "size": 64 * 32 * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        
        # Before entering attention, overwrite mrf[0] and mrf[1] with full HIDDEN_STATES and rmsnorm(HIDDEN_STATES).
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 0,
                "base": HIDDEN_STATES_BASE,
                "size": HIDDEN_STATES_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 1,
                "base": SCRATCH_BASE,
                "size": HIDDEN_STATES_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        
        # [attention] - Load W_Q and W_K
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": QUERY_WEIGHT_BASE,
                "size": QUERY_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 1,
                "base": KEY_WEIGHT_BASE,
                "size": KEY_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),

        # [attention] - Compute Q and K
        Instruction(mnemonic="vzero", args={"vrd": 16}),
        Instruction(mnemonic="vzero", args={"vrd": 17}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 16, "rs1": 1, "rs2": 0}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 17, "rs1": 1, "rs2": 1}),

        # [attention] - Transpose K
        Instruction(mnemonic="vtrpose", args={"vrd": 18, "vs1": 17, "imm": 0}),
        Instruction(mnemonic="vtrpose", args={"vrd": 19, "vs1": 17, "imm": 1}),
        Instruction(mnemonic="vtrpose", args={"vrd": 20, "vs1": 17, "imm": 2}),
        Instruction(mnemonic="vtrpose", args={"vrd": 21, "vs1": 17, "imm": 3}),
        
        # [attention] - quantize Q and K by a scaling factor β (#TODO)
        Instruction(mnemonic="vcast.down", args={"vrd": 22, "vs1": 16}),
        Instruction(mnemonic="vcast.down", args={"vrd": 23, "vs1": 18}),
        Instruction(mnemonic="vcast.down", args={"vrd": 24, "vs1": 19}),
        Instruction(mnemonic="vcast.down", args={"vrd": 25, "vs1": 20}),
        Instruction(mnemonic="vcast.down", args={"vrd": 26, "vs1": 21}),

        # [attention] - Compute Q @ K.T
        # Results are scattered across 4 different registers.
        Instruction(mnemonic="mv.mw.mxu0", args={"rd": 0, "rs1": 23}),
        Instruction(mnemonic="mv.mw.mxu0", args={"rd": 1, "rs1": 24}),
        Instruction(mnemonic="mv.mw.mxu1", args={"rd": 0, "rs1": 25}),
        Instruction(mnemonic="mv.mw.mxu1", args={"rd": 1, "rs1": 26}),
        Instruction(mnemonic="vzero", args={"vrd": 17}),
        Instruction(mnemonic="vzero", args={"vrd": 18}),
        Instruction(mnemonic="vzero", args={"vrd": 19}),
        Instruction(mnemonic="vzero", args={"vrd": 20}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 17, "rs1": 22, "rs2": 0}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 18, "rs1": 22, "rs2": 1}),
        Instruction(mnemonic="matmul.mxu1", args={"rd": 19, "rs1": 22, "rs2": 0}),
        Instruction(mnemonic="matmul.mxu1", args={"rd": 20, "rs1": 22, "rs2": 1}),
        
        # [attention] - Multiply by SCALE = 1 / (sqrt(DIM_HEAD))
        Instruction(
            mnemonic="dma.load",
            args={
                "rd": 4,
                "base": SCALING_BASE,
                "size": SCALING_DATA.numel() * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="vmul", args={"vrd": 21, "vs1": 17, "vs2": 4}),
        Instruction(mnemonic="vmul", args={"vrd": 22, "vs1": 18, "vs2": 4}),
        Instruction(mnemonic="vmul", args={"vrd": 23, "vs1": 19, "vs2": 4}),
        Instruction(mnemonic="vmul", args={"vrd": 24, "vs1": 20, "vs2": 4}),
        
        # [attention] - Find row-wise max
        Instruction(mnemonic="vrowmax", args={"vrd": 25, "vs1": 21}),
        Instruction(mnemonic="vrowmax", args={"vrd": 26, "vs1": 22}),
        Instruction(mnemonic="vrowmax", args={"vrd": 27, "vs1": 23}),
        Instruction(mnemonic="vrowmax", args={"vrd": 28, "vs1": 24}),
        Instruction(mnemonic="vmax", args={"vrd": 29, "vs1": 25, "vs2": 26}),
        Instruction(mnemonic="vmax", args={"vrd": 30, "vs1": 27, "vs2": 28}),
        Instruction(mnemonic="vmax", args={"vrd": 31, "vs1": 29, "vs2": 30}),
        
        # [attention] - Subtract row-wise max from scores
        Instruction(mnemonic="vsub", args={"vrd": 16, "vs1": 21, "vs2": 31}),
        Instruction(mnemonic="vsub", args={"vrd": 17, "vs1": 22, "vs2": 31}),
        Instruction(mnemonic="vsub", args={"vrd": 18, "vs1": 23, "vs2": 31}),
        Instruction(mnemonic="vsub", args={"vrd": 19, "vs1": 24, "vs2": 31}),
        
        # [attention] - Exponentiate each element
        Instruction(mnemonic="vexp", args={"vrd": 20, "vs1": 16}),
        Instruction(mnemonic="vexp", args={"vrd": 21, "vs1": 17}),
        Instruction(mnemonic="vexp", args={"vrd": 22, "vs1": 18}),
        Instruction(mnemonic="vexp", args={"vrd": 23, "vs1": 19}),
        
        # [attention] - Compute row-wise summation
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 24, "vs1": 20}),
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 25, "vs1": 21}),
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 26, "vs1": 22}),
        Instruction(mnemonic="vrot.reduce.sum", args={"vrd": 27, "vs1": 23}),
        Instruction(mnemonic="vadd", args={"vrd": 28, "vs1": 24, "vs2": 25}),
        Instruction(mnemonic="vadd", args={"vrd": 29, "vs1": 26, "vs2": 27}),
        Instruction(mnemonic="vadd", args={"vrd": 30, "vs1": 28, "vs2": 29}),
        
        # [attention] - Divide each element by the sum
        Instruction(mnemonic="vrcp", args={"vrd": 31, "vs1": 30}),
        Instruction(mnemonic="vmul", args={"vrd": 16, "vs1": 20, "vs2": 31}),
        Instruction(mnemonic="vmul", args={"vrd": 17, "vs1": 21, "vs2": 31}),
        Instruction(mnemonic="vmul", args={"vrd": 18, "vs1": 22, "vs2": 31}),
        Instruction(mnemonic="vmul", args={"vrd": 19, "vs1": 23, "vs2": 31}),

        # quantize scores by a scaling factor ζ (#TODO)
        Instruction(mnemonic="vcast.down", args={"vrd": 28, "vs1": 16}),
        Instruction(mnemonic="vcast.down", args={"vrd": 29, "vs1": 17}),
        Instruction(mnemonic="vcast.down", args={"vrd": 30, "vs1": 18}),
        Instruction(mnemonic="vcast.down", args={"vrd": 31, "vs1": 19}),
        
        # [attention] - Load W_V
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": VALUE_WEIGHT_BASE,
                "size": VALUE_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # [attention] - Compute V
        Instruction(mnemonic="vzero", args={"vrd": 20}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 20, "rs1": 1, "rs2": 0}),

        # [attention] - quantize V by a scaling factor γ (#TODO)
        Instruction(mnemonic="vcast.down", args={"vrd": 21, "vs1": 20}),

        # [attention] - Store V to SCRATCH_BASE
        # This is to process 4 different chunks and reduce over for the final attention result.
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 21,
                "base": SCRATCH_BASE,
                "size": 64 * 32 * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # [attention] - Compute scores @ V
        # V is stored as (64, 32) fp8 at SCRATCH_BASE with data in first 16 cols
        # Split into 4 chunks of 16 rows each
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 0,
                "base": SCRATCH_BASE,
                "size": VALUE_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu0",
            args={
                "rd": 1,
                "base": SCRATCH_BASE + 0x0200,
                "size": VALUE_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),

        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 0,
                "base": SCRATCH_BASE + 0x0400,
                "size": VALUE_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 0,
            },
        ),
        Instruction(
            mnemonic="dma.load.mxu1",
            args={
                "rd": 1,
                "base": SCRATCH_BASE + 0x0600,
                "size": VALUE_WEIGHT_DATA.numel() * torch.float8_e4m3fn.itemsize,
                "flag": 1,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),

        Instruction(mnemonic="vzero", args={"vrd": 20}),
        Instruction(mnemonic="vzero", args={"vrd": 21}),
        Instruction(mnemonic="vzero", args={"vrd": 22}),
        Instruction(mnemonic="vzero", args={"vrd": 23}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 20, "rs1": 28, "rs2": 0}),
        Instruction(mnemonic="matmul.mxu0", args={"rd": 21, "rs1": 29, "rs2": 1}),
        Instruction(mnemonic="matmul.mxu1", args={"rd": 22, "rs1": 30, "rs2": 0}),
        Instruction(mnemonic="matmul.mxu1", args={"rd": 23, "rs1": 31, "rs2": 1}),
        Instruction(mnemonic="vadd", args={"vrd": 24, "vs1": 20, "vs2": 21}),
        Instruction(mnemonic="vadd", args={"vrd": 25, "vs1": 22, "vs2": 23}),
        Instruction(mnemonic="vadd", args={"vrd": 26, "vs1": 24, "vs2": 25}),

        # [attention] - Store result WITHOUT quantization (store as bf16, test will handle conversion)
        Instruction(
            mnemonic="dma.store",
            args={
                "rs1": 26,
                "base": SCRATCH_BASE,
                "size": 64 * 16 * torch.bfloat16.itemsize,
                "flag": 0,
            },
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),

        # TODO: Might need strided/gather support from DMA to make residual implementation (or other future operations) easier.
        # Check with tapeout.
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (HIDDEN_STATES_BASE, HIDDEN_STATES_DATA),
        (QUERY_WEIGHT_BASE, QUERY_WEIGHT_DATA),
        (KEY_WEIGHT_BASE, KEY_WEIGHT_DATA),
        (VALUE_WEIGHT_BASE, VALUE_WEIGHT_DATA),
        (OUT_PROJ_WEIGHT_BASE, OUT_PROJ_WEIGHT_DATA),
        (GATE_WEIGHT_BASE, GATE_WEIGHT_DATA),
        (UP_WEIGHT_BASE, UP_WEIGHT_DATA),
        (DOWN_WEIGHT_BASE, DOWN_WEIGHT_DATA),
        (EPS_BASE, EPS_DATA),
        (DIVISOR_BASE, DIVISOR_DATA),
        (SCALING_BASE, SCALING_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        SCRATCH_BASE,
        gemma_layer_generic_fp8(
            HIDDEN_STATES_DATA,
            QUERY_WEIGHT_DATA,
            KEY_WEIGHT_DATA,
            VALUE_WEIGHT_DATA,
            OUT_PROJ_WEIGHT_DATA,
            GATE_WEIGHT_DATA,
            UP_WEIGHT_DATA,
            DOWN_WEIGHT_DATA,
        ).to(torch.bfloat16),
    )
