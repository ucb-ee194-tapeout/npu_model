from typing import List, Tuple, Any
from ...software import Instruction, Program
import torch
from npu_model.isa import (
    DmaArgs,
    ScalarArgs,
    VectorArgs,
)

# Memory layout (DRAM is program-loaded; VMEM is scratchpad accessed by vload/vstore)
DRAM_INPUT_BASE = 0x0000
DRAM_OUTPUT_BASE = 0x0800
VMEM_INPUT_BASE = 0x2000
VMEM_OUTPUT_BASE = 0x2800

# Keep the cube input range modest so BF16 stays numerically well-behaved.
INPUT = torch.linspace(-4.0, 4.0, steps=32 * 32, dtype=torch.bfloat16).reshape(32, 32)


def _bf16_arithmetic_reference(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bfloat16)
    identity = ((x + x).to(torch.bfloat16) - x).to(torch.bfloat16)
    square = (identity * identity).to(torch.bfloat16)
    cube = (x * x * x).to(torch.bfloat16)
    return (square * cube).to(torch.bfloat16)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: List[Instruction[Any]] = [
        # Set up base addresses and transfer size (bytes)
        Instruction(mnemonic="lui", args=ScalarArgs(rd=1, imm=0x2)),  # 0x2000
        Instruction(mnemonic="lui", args=ScalarArgs(rd=2, imm=0x3)),  # 0x2800 = 0x3000 - 0x800
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=2, imm=-2048)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_INPUT_BASE)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=4, imm=0x1)),  # 0x0800 = 0x1000 - 0x800
        Instruction(mnemonic="addi", args=ScalarArgs(rd=4, rs1=4, imm=-2048)),
        Instruction(mnemonic="lui", args=ScalarArgs(rd=5, imm=0x1)),  # 2048 bytes
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=5, imm=-2048)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0),
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        # VMEM -> MRF, compute, MRF -> VMEM
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=1, imm12=32)),
        Instruction("delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=2, vs1=0, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=4, vs1=2, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vsquare.bf16", args=VectorArgs(vd=6, vs1=4)),
        Instruction("delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vcube.bf16", args=VectorArgs(vd=8, vs1=0)),
        Instruction("delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=10, vs1=6, vs2=8)),
        Instruction("delay", args=ScalarArgs(imm=66)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=10, rs1=2, imm12=0)),
        Instruction("delay", args=ScalarArgs(imm=34)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=11, rs1=2, imm12=32)),
        # Ensure the VPU has time to commit the VMEM write before DMA reads it.
        Instruction("delay", args=ScalarArgs(imm=34)),
        # VMEM -> DRAM
        Instruction(
            mnemonic="dma.store.ch<N>",
            args=DmaArgs(rd=4, rs1=2, rs2=5, channel=0),
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_INPUT_BASE, INPUT),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        _bf16_arithmetic_reference(INPUT),
    )
