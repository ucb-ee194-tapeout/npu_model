from typing import List, Tuple, Any
import math
import torch
from ...software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs


# NOTE: This program is currently written for a single 32x16 tile.
# Use SEQ_LEN=32 so Q @ K produces one bf16 tile (32x16) in this model.
SEQ_LEN = 32
HEAD_DIM = 16

# Data tensors in fp8 sized to one 32x32 tile (what MXU expects).
# We encode logical (SEQ_LEN x HEAD_DIM) by zero-padding within the tile:
# - Q: ones in first 16 cols, zeros elsewhere
# - K: ones in first 16 rows/cols block, zeros elsewhere
QUERY_DATA = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
QUERY_DATA[:, :HEAD_DIM] = torch.ones((32, HEAD_DIM), dtype=torch.float8_e4m3fn)
KEY_DATA = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
KEY_DATA[:HEAD_DIM, :HEAD_DIM] = torch.ones((HEAD_DIM, HEAD_DIM), dtype=torch.float8_e4m3fn)

# Scaling matrix: every entry is 1 / sqrt(HEAD_DIM), in bf16 for vector ops
SCALE_VALUE = 1.0 / math.sqrt(float(HEAD_DIM))
SCALE_DATA = torch.full((SEQ_LEN, HEAD_DIM), SCALE_VALUE, dtype=torch.bfloat16)

# DRAM layout (program-loaded)
DRAM_QUERY_BASE = 0x0000
DRAM_KEY_BASE = 0x0400
DRAM_SCALE_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x0C00

# VMEM layout
VMEM_QUERY_BASE = 0x2000
VMEM_KEY_BASE = 0x2400
VMEM_SCALE_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x2C00


class GemmaAttentionProgram(Program):
    """
    Gemma attention kernel program (simplified, single-head).

    This program demonstrates a scaled dot-product attention block using
    the NPU ISA:
      - `matmul.mxu0` for Q @ K and softmax(QK^T) @ V
      - `vexp`, `vreduce.sum`, `vrcp`, and `vmul` to implement softmax.
    """

    instructions: List[Instruction[Any]] = [
        # Register setup (VMEM) (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        # 0x2400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x2)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=0x400)),
        # 0x2800 = 0x3000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-2048)),
        # 0x2C00 = 0x3000 - 0x400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=-1024)),
        # Register setup (DRAM)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=6, rs1=0, imm=DRAM_QUERY_BASE)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=0, imm=DRAM_KEY_BASE)),
        # DRAM_SCALE_BASE = 0x0800 = 0x1000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=9, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=9, rs1=9, imm=-2048)),
        # DRAM_OUTPUT_BASE = 0x0C00 = 0x1000 - 0x400
        Instruction(mnemonic="lui", args=ScalarArgs(rd=10, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=10, rs1=10, imm=-1024)),
        # Byte lengths: fp8 tile (1024) and bf16 tile (1024)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=11, rs1=0, imm=1024)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=12, rs1=0, imm=1024)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=6, rs2=11, channel=0)
        ),  # Q tile
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=7, rs2=11, channel=1)
        ),  # K tile
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=4, rs1=9, rs2=12, channel=2)
        ),  # scale (bf16 tile)
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=2)),
        # VMEM -> MRF
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)
        ),  # Q (fp8 tile)
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=1, rs1=2, imm12=0)
        ),  # K (fp8 tile)
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=2, rs1=4, imm12=0)
        ),  # scale (bf16 tile, 32x16)
        # Push K to WB slot 0, compute scores = Q @ K, pop bf16
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=1)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        Instruction(
            mnemonic="vmatpop.bf16.acc.mxu0", args=VectorArgs(vd=3, vs1=0)
        ),  # scores bf16 tile
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        # scores_scaled = scores * scale
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=3, vs2=2)),
        # Softmax (unnormalized variant: no max subtraction)
        # exp_scores = exp(scores_scaled)
        Instruction(mnemonic="vexp.bf16", args=VectorArgs(vd=5, vs1=4)),
        # row_sum = sum(exp_scores) broadcast across columns
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=6, vs1=5)),
        # inv_row_sum = 1 / row_sum
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=7, vs1=6)),
        # softmax_scores = exp_scores * inv_row_sum
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=8, vs1=5, vs2=7)),
        # Store softmax scores (bf16 tile)
        Instruction(mnemonic="vstore", args=VectorArgs(vd=8, rs1=5, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=10, rs1=5, rs2=12, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_QUERY_BASE, QUERY_DATA),
        (DRAM_KEY_BASE, KEY_DATA),
        (DRAM_SCALE_BASE, SCALE_DATA),
    ]

    # Golden result: softmax(scores_scaled) (no max subtraction), pure torch
    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.softmax(
            ((QUERY_DATA.to(torch.float32) @ KEY_DATA.to(torch.float32)) * SCALE_VALUE)[
                :, :HEAD_DIM
            ],
            dim=1,
        ).to(torch.bfloat16),
    )
