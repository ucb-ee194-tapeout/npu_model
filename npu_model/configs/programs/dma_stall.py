from typing import List, Tuple, Any
from npu_model.software import Instruction, Program
from npu_model.isa import DmaArgs, MatrixArgs, VectorArgs, ScalarArgs
import torch


class DMAStallProgram(Program):
    """
    A simple program demonstrating DMA loads, stalling logic, and matrix multiplication
    updated for the latest npu_model ISA.
    """

    instructions: List[Instruction[Any]] = [
        # --- 1. Setup Scalar Registers ---
        # Set x2 = 1024 (Base Address & Size 1024)
        Instruction(mnemonic="addi", args=ScalarArgs(rd=1, rs1=0, imm=0)),
        Instruction(mnemonic="addi", args=ScalarArgs(rd=2, rs1=0, imm=1024)),
        # --- 2. Configure DMA Channels ---
        # Configure Base Address x1 (0)
        Instruction(mnemonic="dma.config.ch<N>", args=DmaArgs(rs1=0, channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        # --- 3. Load thing ---
        # Load 1024 bytes (x2) from DRAM to VMEM(x1) on Channel 0
        # Note: rs1 specifies VMEM offset, rs2 specifies length. Base address comes from dma.config
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=0, rs1=1, rs2=2, channel=0)
        ),
        # Load 1024 bytes (x2) from DRAM to VMEM(x2) on Channel 1
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=1, rs1=2, rs2=2, channel=1)
        ),
        # Wait to get these things in VMEM
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # Move VMEM data to actual computational registers
        # vload VMEM(x1=0) -> MRF 2
        Instruction(mnemonic="vload", args=VectorArgs(vd=1, rs1=0, imm12=0)),
        # vload VMEM(x2=1024) -> Temporary MRF 1
        Instruction(mnemonic="vload", args=VectorArgs(vd=0, rs1=1, imm12=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=16)),
        # Push Temporary MRF 1 -> MXU0 Weight Buffer 1
        Instruction(mnemonic="vmatpush.weight.mxu0", args=VectorArgs(vd=0, vs1=0)),
        # --- 4. Do unnecessary loads (Overlapped with Matmul) ---
        # We do not need to reconfigure the DMA channels if the base addresses are unchanged
        # Issue DMA loads again (DRAM -> VMEM)
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=3, rs1=0, rs2=1, channel=0)
        ),
        Instruction(
            mnemonic="dma.load.ch<N>", args=DmaArgs(rd=4, rs1=1, rs2=1, channel=1)
        ),
        # --- 5. Do matmul ---
        Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=1, vs2=0)),
        Instruction(mnemonic="delay", args=ScalarArgs(imm=32)), # TODO - verify delays
        # Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        # Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        # Instruction(mnemonic="vmatmul.mxu0", args=MatrixArgs(vd=0, vs1=0, vs2=0)),
        Instruction(mnemonic="vmatpop.fp8.acc.mxu0", args=VectorArgs(vd=0, vs1=0)),
        # Instruction(mnemonic="delay", args=ScalarArgs(imm=32)),
        # Wait to finish unnecessary loads
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=0)),
        Instruction(mnemonic="dma.wait.ch<N>", args=DmaArgs(channel=1)),
        # End delay
        Instruction(mnemonic="delay", args=ScalarArgs(imm=0)),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (0, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
        (1024, torch.eye(32, 32, dtype=torch.float8_e4m3fn)),
    ]
