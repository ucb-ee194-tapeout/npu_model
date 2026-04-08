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
DRAM_OUTPUT_BASE = 0x0400
VMEM_INPUT_BASE = 0x2000
VMEM_OUTPUT_BASE = 0x2400

# One MRF register worth of bf16 data: (mrf_depth, mrf_width / bf16_bytes)
# With default configs this is typically 32x16 elements = 1024 bytes.
INPUT = torch.arange(32 * 16, dtype=torch.bfloat16).reshape(32, 16)


class VectorArithmeticProgram(Program):
    """
    Basic arithmetic correctness
    """

    instructions: List[Instruction[Any]] = [
        # Set up base addresses and transfer size (bytes)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=VMEM_INPUT_BASE)),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=VMEM_OUTPUT_BASE)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=3, rs1=0, imm=DRAM_INPUT_BASE)),
        Instruction(
            mnemonic="addi", args=ScalarArgs(rd=4, rs1=0, imm=DRAM_OUTPUT_BASE)
        ),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=5, rs1=0, imm=1024)),
        # DRAM -> VMEM
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(
            mnemonic="dma.load.ch<N>",
            args=DmaArgs(rd=1, rs1=3, rs2=5, channel=0),
        ),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        # VMEM -> MRF, compute, MRF -> VMEM
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction("delay", args=ScalarArgs(imm=16)),
        Instruction(mnemonic="vadd.bf16", args=VectorArgs(vd=1, vs1=0, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vsub.bf16", args=VectorArgs(vd=2, vs1=1, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vmul.bf16", args=VectorArgs(vd=3, vs1=2, vs2=0)),
        Instruction("delay", args=ScalarArgs(imm=32)),
        Instruction(mnemonic="vstore", args=VectorArgs(vd=3, rs1=2, imm12=0)),
        # Ensure the VPU has time to commit the VMEM write before DMA reads it.
        # There is currently no explicit VPU↔DMA memory ordering primitive in the model.
        Instruction("delay", args=ScalarArgs(imm=16)),
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
        (INPUT**2),
    )
