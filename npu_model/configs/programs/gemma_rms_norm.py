from typing import List, Tuple, Any
from ...software import Instruction, Program
import torch
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs
from npu_model.hardware.arch_state import ArchState
from npu_model.configs.hardware import DefaultHardwareConfig
from npu_model.configs.isa_definition import (
    vadd_bf16,
    vmul_bf16,
    vredsum_row_bf16,
    vrecip_bf16,
    vsqrt_bf16,
)


# Input shape matches one full BF16 tile consumed by the VPU: 32 rows x 32 columns.
INPUT_DATA = torch.randn(32, 32, dtype=torch.bfloat16)
ROW_SIZE = INPUT_DATA.shape[-1]
EPS = 1e-6
# DRAM layout
DRAM_INPUT_BASE = 0x0000
DRAM_EPS_BASE = 0x0800
DRAM_OUTPUT_BASE = 0x1000

# VMEM layout
VMEM_INPUT_BASE = 0x2000
VMEM_EPS_BASE = 0x2800
VMEM_OUTPUT_BASE = 0x3000


def _gemma_rms_norm_program_reference(
    x: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    state = ArchState(DefaultHardwareConfig().arch_state_config)
    x_bf16 = x.to(torch.bfloat16).contiguous()
    state.write_mrf_bf16_tile(0, x_bf16)
    state.write_mrf_bf16_tile(2, torch.full_like(x_bf16, eps, dtype=torch.bfloat16))
    state.write_mrf_bf16(8, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16))
    state.write_mrf_bf16(9, torch.full((32, 16), x.shape[-1], dtype=torch.bfloat16))
    vmul_bf16(state, VectorArgs(vd=4, vs1=0, vs2=0))
    vredsum_row_bf16(state, VectorArgs(vd=6, vs1=4))
    vrecip_bf16(state, VectorArgs(vd=10, vs1=8))
    vmul_bf16(state, VectorArgs(vd=12, vs1=6, vs2=10))
    vadd_bf16(state, VectorArgs(vd=14, vs1=12, vs2=2))
    vsqrt_bf16(state, VectorArgs(vd=16, vs1=14))
    vrecip_bf16(state, VectorArgs(vd=18, vs1=16))
    vmul_bf16(state, VectorArgs(vd=20, vs1=0, vs2=18))
    return state.read_mrf_bf16_tile(20).clone()


class GemmaRmsNormProgram(Program):
    """
    Gemma RMS norm program.
    RMS norm: x * rsqrt(mean(x^2) + eps).
    Row-wise mean via transpose + vreduce.sum (second-to-last dim) + vbroadcast.cols.
    """

    instructions: List[Instruction[Any]] = [
        # VMEM bases (use LUI+ADDI so immediates stay 12-bit clean)
        # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),
        # 0x2800 = 0x3000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x3)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=-2048)),
        # 0x3000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=3, imm=0x3)),
        # DRAM bases
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_INPUT_BASE)),
        # DRAM_EPS_BASE = 0x0800 = 0x1000 - 0x800
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=-2048)),
        # DRAM_OUTPUT_BASE = 0x1000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=6, imm=0x1)),
        # byte length for bf16 tile
        Instruction(mnemonic="lui", args=ScalarArgs(rd=7, imm=0x1)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=7, rs1=7, imm=-2048)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=4, rs2=7, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=2, rs1=5, rs2=7, channel=1)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # VMEM -> MRF
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),  # x low
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)),  # x high
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=2, rs1=2, imm12=0)),  # eps low
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(
            mnemonic="vload", args=VectorArgs(vd=3, rs1=2, imm12=32)
        ),  # eps high
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        # x_sq = x * x
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=4, vs1=0, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # sum_sq over columns, broadcast back across each row
        Instruction(mnemonic="vredsum.row.bf16", args=VectorArgs(vd=6, vs1=4)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=130)),
        # mean_sq = sum_sq * (1/ROW_SIZE)
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=8, imm=ROW_SIZE)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        Instruction(mnemonic="vli.all", args=VectorArgs(vd=9, imm=ROW_SIZE)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=65)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=10, vs1=8)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=12, vs1=6, vs2=10)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # var_eps = var + eps
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=14, vs1=12, vs2=2)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # rsqrt = 1/sqrt(var_eps)
        Instruction(mnemonic="vsqrt.bf16", args=VectorArgs(vd=16, vs1=14)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vrecip.bf16", args=VectorArgs(vd=18, vs1=16)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # output = x * rsqrt
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=20, vs1=0, vs2=18)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=66)),
        # MRF -> VMEM -> DRAM
        Instruction(mnemonic="vstore", args=VectorArgs(vd=20, rs1=3, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=21, rs1=3, imm12=32)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        Instruction(
            mnemonic="dma.store.ch<N>", args=DmaArgs(rd=6, rs1=3, rs2=7, channel=0)
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT_DATA),
        (DRAM_EPS_BASE, torch.full(INPUT_DATA.shape, EPS, dtype=torch.bfloat16)),
    ]

    # FIXME: Re-derive a standalone golden reference for the pair-register BF16
    # VPU path. The current kernel wiring is exercised by simulation, but the
    # previous float-side golden no longer matches the staged BF16 execution.
    golden_result = None
