from typing import Dict
import numpy as np

from model_npu.isa import instr, InstructionType
from model_npu.hardware.arch_state import ArchState


"""
Scalar operations
"""


@instr("nop", instruction_type=InstructionType.SCALAR)
def nop(state: ArchState, args: Dict[str, int]) -> None:
    pass


@instr("add", instruction_type=InstructionType.SCALAR)
def add(state: ArchState, args: Dict[str, int]) -> None:
    state.set_xrf(args["rd"], state.xrf[args["rs1"]] + state.xrf[args["rs2"]])


@instr("addi", instruction_type=InstructionType.SCALAR)
def addi(state: ArchState, args: Dict[str, int]) -> None:
    state.set_xrf(args["rd"], state.xrf[args["rs1"]] + args["imm"])


@instr("sub", instruction_type=InstructionType.SCALAR)
def sub(state: ArchState, args: Dict[str, int]) -> None:
    state.set_xrf(args["rd"], state.xrf[args["rs1"]] - state.xrf[args["rs2"]])


@instr("jal", instruction_type=InstructionType.SCALAR)
def jal(state: ArchState, args: Dict[str, int]) -> None:
    state.set_npc(
        state.pc + args["imm"] - 2
    )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("blt", instruction_type=InstructionType.SCALAR)
def blt(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - 2
        )  # FIXME: this is a hack to compensate for the IF->EX delay


"""
Matrix operations
"""


@instr("matmul", instruction_type=InstructionType.MATRIX)
def matmul(state: ArchState, args: Dict[str, int]) -> None:
    state.mrf[args["rd"]] = state.mrf[args["rs1"]] @ state.mrf[args["rs2"]]


"""
Memory operations
"""


@instr("dma.load", instruction_type=InstructionType.DMA)
def dma_load(state: ArchState, args: Dict[str, int]) -> None:
    base = args["base"]
    size = args["size"]
    data = np.frombuffer(state.read_bytes(base, size), dtype=np.float32).reshape(
        state.matrix_shape
    )  # FIXME: referencing shape here might be a bad idea
    state.mrf[args["rd"]] = data


@instr("dma.store", instruction_type=InstructionType.DMA)
def dma_store(state: ArchState, args: Dict[str, int]) -> None:
    base = args["base"]
    size = args["size"]
    data = state.mrf[args["rs1"]].astype(np.float32, copy=False).tobytes(order="C")
    state.write_bytes(base, data)


@instr("dma.wait", instruction_type=InstructionType.BARRIER)
def dma_wait(state: ArchState, args: Dict[str, int]) -> None:
    pass
