from typing import Dict
import torch

from model_npu.isa import instr, InstructionType
from model_npu.hardware.arch_state import ArchState


"""
Scalar operations
"""


@instr("delay", instruction_type=InstructionType.SCALAR)
def delay(state: ArchState, args: Dict[str, int]) -> None:
    """
    Delay for a specified number of cycles.

    a `delay 0` is equivalent to a `nop` operation.
    """
    # delay is handled at IDU stage
    pass


@instr("addi", instruction_type=InstructionType.SCALAR)
def addi(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + args["imm"])


@instr("slli", instruction_type=InstructionType.SCALAR)
def slli(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] << args["imm"])


@instr("slti", instruction_type=InstructionType.SCALAR)
def slti(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], 1 if state.xrf[args["rs1"]] < args["imm"] else 0)


@instr("sltiu", instruction_type=InstructionType.SCALAR)
def sltiu(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], 1 if state.xrf[args["rs1"]] < args["imm"] else 0)


@instr("xori", instruction_type=InstructionType.SCALAR)
def xori(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] ^ args["imm"])


@instr("srli", instruction_type=InstructionType.SCALAR)
def srli(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> args["imm"])


@instr("srai", instruction_type=InstructionType.SCALAR)
def srai(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> args["imm"])


@instr("ori", instruction_type=InstructionType.SCALAR)
def ori(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] | args["imm"])


@instr("andi", instruction_type=InstructionType.SCALAR)
def andi(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] & args["imm"])


@instr("add", instruction_type=InstructionType.SCALAR)
def add(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] + state.xrf[args["rs2"]])


@instr("sub", instruction_type=InstructionType.SCALAR)
def sub(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] - state.xrf[args["rs2"]])


@instr("sll", instruction_type=InstructionType.SCALAR)
def sll(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] << state.xrf[args["rs2"]])


@instr("slt", instruction_type=InstructionType.SCALAR)
def slt(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(
        args["rd"], 1 if state.xrf[args["rs1"]] < state.xrf[args["rs2"]] else 0
    )


@instr("sltu", instruction_type=InstructionType.SCALAR)
def sltu(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(
        args["rd"], 1 if state.xrf[args["rs1"]] < state.xrf[args["rs2"]] else 0
    )  # TODO: sign


@instr("xor", instruction_type=InstructionType.SCALAR)
def xor(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] ^ state.xrf[args["rs2"]])


@instr("srl", instruction_type=InstructionType.SCALAR)
def srl(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> state.xrf[args["rs2"]])


@instr("sra", instruction_type=InstructionType.SCALAR)
def sra(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] >> state.xrf[args["rs2"]])


@instr("or", instruction_type=InstructionType.SCALAR)
def or_(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] | state.xrf[args["rs2"]])


@instr("and", instruction_type=InstructionType.SCALAR)
def and_(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["rs1"]] & state.xrf[args["rs2"]])


@instr("lui", instruction_type=InstructionType.SCALAR)
def lui(state: ArchState, args: Dict[str, int]) -> None:
    state.write_xrf(args["rd"], state.xrf[args["imm"]] << 12)


PIPELINE_LATENCY = 2


@instr("jal", instruction_type=InstructionType.SCALAR)
def jal(state: ArchState, args: Dict[str, int]) -> None:
    state.set_npc(
        state.pc + args["imm"] - PIPELINE_LATENCY
    )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("beq", instruction_type=InstructionType.SCALAR)
def beq(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] == state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bne", instruction_type=InstructionType.SCALAR)
def bne(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] != state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("blt", instruction_type=InstructionType.SCALAR)
def blt(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bge", instruction_type=InstructionType.SCALAR)
def bge(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bltu", instruction_type=InstructionType.SCALAR)
def bltu(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bgeu", instruction_type=InstructionType.SCALAR)
def bgeu(state: ArchState, args: Dict[str, int]) -> None:
    if state.xrf[args["rs1"]] < state.xrf[args["rs2"]]:
        state.set_npc(
            state.pc + args["imm"] - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


"""
Matrix operations
"""


@instr("mv.mw", instruction_type=InstructionType.MATRIX)
def mv_mw(state: ArchState, args: Dict[str, int]) -> None:
    """
    Vector/matrix move from matrix registers to weight buffer.
    """
    # TODO: check register dimensions
    state.write_wb_bf16(args["rd"], state.read_mrf_bf16(args["rs1"]))


@instr("matmul.mxu0", instruction_type=InstructionType.MATRIX)
def matmul_mxu0(state: ArchState, args: Dict[str, int]) -> None:
    """
    Matrix multiplication using MXU0, the systolic array.
    """
    activation = state.read_mrf_bf16(args["rs1"])
    weight = state.read_wb_bf16(args["rs2"])
    accumulation = (activation @ weight.T).to(torch.float32)
    state.write_mrf_f32(args["rd"], accumulation)


@instr("matmul.mxu1", instruction_type=InstructionType.MATRIX)
def matmul_mxu1(state: ArchState, args: Dict[str, int]) -> None:
    """
    Matrix multiplication using MXU1, the parallel inner product tree.
    """
    activation = state.read_mrf_bf16(args["rs1"])
    weight = state.read_wb_bf16(args["rs2"])
    accumulation = (activation @ weight.T).to(torch.float32)
    state.write_mrf_f32(args["rd"], accumulation)


"""
Vector operations (bfloat16)
"""


@instr("vadd", instruction_type=InstructionType.VECTOR)
def vadd(state: ArchState, args: Dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    state.write_mrf_bf16(args["vrd"], (a + b).to(torch.bfloat16))


@instr("vsub", instruction_type=InstructionType.VECTOR)
def vsub(state: ArchState, args: Dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    state.write_mrf_bf16(args["vrd"], (a - b).to(torch.bfloat16))


@instr("vmul", instruction_type=InstructionType.VECTOR)
def vmul(state: ArchState, args: Dict[str, int]) -> None:
    a = state.read_mrf_bf16(args["vs1"])
    b = state.read_mrf_bf16(args["vs2"])
    result = (a * b).to(torch.bfloat16)
    print(result)
    state.write_mrf_bf16(args["vrd"], (a * b).to(torch.bfloat16))


@instr("vsqrt", instruction_type=InstructionType.VECTOR)
def vsqrt(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.sqrt(x).to(torch.bfloat16))


@instr("vrcp", instruction_type=InstructionType.VECTOR)
def vrcp(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], (1.0 / x).to(torch.bfloat16))


@instr("vexp", instruction_type=InstructionType.VECTOR)
def vexp(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.exp(x).to(torch.bfloat16))


@instr("vlog2", instruction_type=InstructionType.VECTOR)
def vlog2(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.log2(x).to(torch.bfloat16))


@instr("vexp2", instruction_type=InstructionType.VECTOR)
def vexp2(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.exp2(x).to(torch.bfloat16))


@instr("vsin", instruction_type=InstructionType.VECTOR)
def vsin(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.sin(x).to(torch.bfloat16))


@instr("vcos", instruction_type=InstructionType.VECTOR)
def vcos(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.cos(x).to(torch.bfloat16))


@instr("vtanh", instruction_type=InstructionType.VECTOR)
def vtanh(state: ArchState, args: Dict[str, int]) -> None:
    x = state.read_mrf_bf16(args["vs1"])
    state.write_mrf_bf16(args["vrd"], torch.tanh(x).to(torch.bfloat16))


@instr("mv.mm", instruction_type=InstructionType.VECTOR)
def mv_mm(state: ArchState, args: Dict[str, int]) -> None:
    """
    Vector/matrix move between matrix registers.
    """
    state.write_mrf_f32(args["rd"], state.read_mrf_f32(args["rs1"]))


"""
Memory operations
"""


@instr("dma.load.m", instruction_type=InstructionType.DMA)
def dma_load_m(state: ArchState, args: Dict[str, int]) -> None:
    """
    DMA load from memory to matrix registers.
    """
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


@instr("dma.load.w", instruction_type=InstructionType.DMA)
def dma_load_w(state: ArchState, args: Dict[str, int]) -> None:
    """
    DMA load from memory to weight buffer.
    """
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


@instr("dma.store.m", instruction_type=InstructionType.DMA)
def dma_store_m(state: ArchState, args: Dict[str, int]) -> None:
    """
    DMA store from matrix registers to memory.
    """
    base = args["base"]
    size = args["size"]
    data = state.mrf[args["rs1"]].view(torch.uint8)
    state.write_memory(base, data[:size])


@instr("dma.wait", instruction_type=InstructionType.BARRIER)
def dma_wait(state: ArchState, args: Dict[str, int]) -> None:
    """
    Wait for target DMA operations to complete.
    """
    pass
