from typing import Dict
import torch

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
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + state.xrf[args["rs2"]])


@instr("addi", instruction_type=InstructionType.SCALAR)
def addi(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + args["imm"])


@instr("sub", instruction_type=InstructionType.SCALAR)
def sub(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] - state.xrf[args["rs2"]])


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
    activation = state.read_mrf_bf16(args["rs1"])
    weight = state.read_wb_bf16(args["rs2"])
    accumulation = (activation @ weight.T).to(torch.float32)
    state.write_mrf_f32(args["rd"], accumulation)


"""
Memory operations
"""


@instr("dma.load", instruction_type=InstructionType.DMA)
def dma_load(state: ArchState, args: Dict[str, int]) -> None:
    base = args["base"]
    size = args["size"]
    data = state.read_memory(base, size)
    # zero pad the data to the size of the MRF
    if data.numel() < state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize:
        data = torch.nn.functional.pad(
            data,
            (
                0,
                state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize
                - data.numel(),
            ),
        )
    state.write_mrf_u8(args["rd"], data)


@instr("dma.loadw", instruction_type=InstructionType.DMA)
def dma_loadw(state: ArchState, args: Dict[str, int]) -> None:
    base = args["base"]
    size = args["size"]
    data = torch.tensor(state.read_memory(base, size), dtype=torch.uint8)
    # zero pad the data to the size of the WB
    if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
        data = torch.nn.functional.pad(
            data,
            (
                0,
                state.cfg.wb_depth * state.cfg.wb_width // torch.uint8.itemsize
                - data.numel(),
            ),
        )
    state.write_wb_u8(args["rd"], data)


@instr("dma.store", instruction_type=InstructionType.DMA)
def dma_store(state: ArchState, args: Dict[str, int]) -> None:
    base = args["base"]
    size = args["size"]
    data = state.mrf[args["rs1"]].view(torch.uint8)
    state.write_memory(base, data[:size])


@instr("dma.wait", instruction_type=InstructionType.BARRIER)
def dma_wait(state: ArchState, args: Dict[str, int]) -> None:
    pass
