from typing import List, Tuple
from ...software import Instruction, Program
import torch


DRAM_ACTIVATION_BASE = 0x0000
DRAM_WEIGHT_BASE = 0x0400
DRAM_OUTPUT_BASE = 0x0800
VMEM_ACTIVATION_BASE = 0x2000
VMEM_WEIGHT_BASE = 0x2400
VMEM_OUTPUT_BASE = 0x2800

ACTIVATION_DATA = torch.eye(32, 32, dtype=torch.float8_e4m3fn)
WEIGHT_DATA = (2 * torch.eye(32, 32, dtype=torch.float32)).to(torch.float8_e4m3fn)
MATMUL_RESULT = (
    ACTIVATION_DATA.to(torch.float32) @ WEIGHT_DATA.to(torch.float32)
).to(torch.bfloat16)


class MatmulProgram(Program):
    """
    matmul test
    """

    instructions: List[Instruction] = [
        Instruction(
            mnemonic="addi", args={"rd": 1, "rs1": 0, "imm": VMEM_ACTIVATION_BASE}
        ),
        Instruction(
            mnemonic="addi", args={"rd": 2, "rs1": 0, "imm": VMEM_WEIGHT_BASE}
        ),
        Instruction(mnemonic="addi", args={"rd": 3, "rs1": 0, "imm": VMEM_OUTPUT_BASE}),
        Instruction(
            mnemonic="dma.load",
            args={"rd": 8, "base": DRAM_ACTIVATION_BASE, "size": 1024, "flag": 0},
        ),
        Instruction(
            mnemonic="dma.load",
            args={"rd": 9, "base": DRAM_WEIGHT_BASE, "size": 1024, "flag": 1},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
        Instruction(mnemonic="vstore", args={"vd": 8, "rs1": 1, "offset": 0}),
        Instruction(mnemonic="vstore", args={"vd": 9, "rs1": 2, "offset": 0}),
        Instruction(mnemonic="vload", args={"vd": 0, "rs1": 1, "offset": 0}),
        Instruction(mnemonic="vload", args={"vd": 1, "rs1": 2, "offset": 0}),
        Instruction(mnemonic="vmatpush.weight.mxu0", args={"vd": 0, "vs1": 1}),
        Instruction(mnemonic="delay", args={"imm": 0}, delay=16),
        Instruction(mnemonic="vmatmul.mxu0", args={"vd": 0, "vs1": 0, "vs2": 0}),
        Instruction(mnemonic="delay", args={"imm": 0}, delay=32),
        Instruction(mnemonic="vmatpop.bf16.acc.mxu0", args={"vd": 2, "vs2": 0}),
        Instruction(mnemonic="vstore", args={"vd": 2, "rs1": 3, "offset": 0}),
        Instruction(mnemonic="vstore", args={"vd": 3, "rs1": 3, "offset": 32}),
        Instruction(mnemonic="vload", args={"vd": 10, "rs1": 3, "offset": 0}),
        Instruction(mnemonic="vload", args={"vd": 11, "rs1": 3, "offset": 32}),
        Instruction(mnemonic="delay", args={"imm": 0}, delay=16),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 10, "base": DRAM_OUTPUT_BASE, "size": 1024, "flag": 0},
        ),
        Instruction(
            mnemonic="dma.store",
            args={"rs1": 11, "base": DRAM_OUTPUT_BASE + 1024, "size": 1024, "flag": 1},
        ),
        Instruction(mnemonic="dma.wait", args={"flag": 0}),
        Instruction(mnemonic="dma.wait", args={"flag": 1}),
    ]

    memory_regions: List[Tuple[int, torch.Tensor]] = [
        (DRAM_ACTIVATION_BASE, ACTIVATION_DATA),
        (DRAM_WEIGHT_BASE, WEIGHT_DATA),
    ]

    golden_result: tuple[int, torch.Tensor] = (
        DRAM_OUTPUT_BASE,
        torch.cat((MATMUL_RESULT[:, :16], MATMUL_RESULT[:, 16:]), dim=0),
    )
