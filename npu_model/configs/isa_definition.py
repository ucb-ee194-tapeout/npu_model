import torch

from typing import Any
from npu_model.hardware import dma
from npu_model.hardware import arch_state
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,
    ScalarArgs,
    VectorArgs,
    instr,
    InstructionType,
)
from npu_model.hardware.arch_state import ArchState
from npu_model.software import instruction


PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Helper Functions
# =============================================================================


def _sign_extend(value: int, length: int):
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length

    return value & 0xFFFFFFFF


def _int_to_le_bytes(data, length) -> torch.Tensor:
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}

    if length not in type_map:
        raise ValueError("Length must be 1, 2, or 4 bytes.")

    return torch.tensor([data], dtype=type_map[length]).view(torch.uint8).clone()


def _le_bytes_to_int(tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}

    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")

    raw_val = tensor.contiguous().view(type_map[length]).item()

    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return raw_val & masks[length]


def _tensor_register_bytes(state: ArchState) -> int:
    return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize


def _vls_base_register(args: VectorArgs) -> int:
    if hasattr(args, "rs1"):
        return args.rs1
    else:
        raise KeyError("vload/vstore requires rs1")


def _vls_offset(args: VectorArgs) -> int:
    if hasattr(args, "offset"):
        return args.offset
    else:
        raise KeyError("no offset provided")


def _vls_address(state: ArchState, args: VectorArgs) -> int:
    return state.read_xrf(_vls_base_register(args)) + (_vls_offset(args) << 5)


def _acc_dest_index(args: VectorArgs) -> int:
    if hasattr(args, "vd"):
        return args.vd
    raise KeyError("MXU local write requires a destination accumulator selector")


def _acc_source_index(args: Any) -> int:
    if hasattr(args, "vs2"):
        return args.vs2
    if hasattr(args, "vs1"):
        return args.vs1
    if hasattr(args, "vd"):
        return args.vd
    raise KeyError("MXU local read requires an accumulator selector")


def _vmatmul(state: ArchState, unit: str, args: MatrixArgs, accumulate: bool) -> None:
    activation_fp16 = state.read_mrf_fp8(args.vs1).to(torch.float16)
    weight_fp16 = state.read_wb_fp8(unit, args.vs2).to(torch.float16)
    result_fp16 = activation_fp16 @ weight_fp16
    if accumulate:
        result_fp16 = result_fp16 + state.read_acc_bf16(unit, _acc_dest_index(args)).to(
            torch.float16
        )
    state.write_acc_bf16(unit, _acc_dest_index(args), result_fp16.to(torch.bfloat16))


# =============================================================================
# Instructions Matching README Specification
# =============================================================================


@instr("lb", instruction_type=InstructionType.I.SCALAR)
def lb(state: ArchState, args: ScalarArgs) -> None:
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), args.imm, 1))
    state.write_xrf(args.rd, _sign_extend(value, 8))


@instr("lh", instruction_type=InstructionType.I.SCALAR)
def lh(state: ArchState, args: ScalarArgs) -> None:
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), args.imm, 2))
    state.write_xrf(args.rd, _sign_extend(value, 16))


@instr("lw", instruction_type=InstructionType.I.SCALAR)
def lw(state: ArchState, args: ScalarArgs) -> None:
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), args.imm, 4))
    state.write_xrf(args.rd, value)


@instr("lbu", instruction_type=InstructionType.I.SCALAR)
def lbu(state: ArchState, args: ScalarArgs) -> None:
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), args.imm, 1))
    state.write_xrf(args.rd, value)


@instr("lhu", instruction_type=InstructionType.I.SCALAR)
def lhu(state: ArchState, args: ScalarArgs) -> None:
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), args.imm, 2))
    state.write_xrf(args.rd, value)


@instr("seld", instruction_type=InstructionType.I.SCALAR)
def seld(state: ArchState, args: ScalarArgs) -> None:
    state.write_erf(args.rd, state.read_vmem(state.read_xrf(args.rs1), args.imm, 1))


@instr("seli", instruction_type=InstructionType.I.SCALAR)
def seli(state: ArchState, args: ScalarArgs):
    state.write_erf(args.rd, args.imm)


@instr("vload", instruction_type=InstructionType.VLS.VECTOR)
def vload(state: ArchState, args: VectorArgs) -> None:
    data = state.read_vmem(
        state.read_xrf(args.rs1), args.imm12 << 5, _tensor_register_bytes(state)
    ).view(
        torch.uint8
    )  # FIX: Use .view() to preserve float8 bit patterns
    state.write_mrf_u8(args.vd, data)


@instr("vstore", instruction_type=InstructionType.VLS.VECTOR)
def vstore(state: ArchState, args: VectorArgs) -> None:
    data = state.mrf[args.vd].view(torch.uint8)
    state.write_vmem(
        state.read_xrf(args.rs1), args.imm12 << 5, data  # FIX: Shift by 5, not 12
    )


@instr("fence", instruction_type=InstructionType.I.SCALAR)
def fence(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr("addi", instruction_type=InstructionType.I.SCALAR)
def addi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + args.imm)


@instr("slli", instruction_type=InstructionType.I.SCALAR)
def slli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << args.imm)


@instr("slti", instruction_type=InstructionType.I.SCALAR)
def slti(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < args.imm else 0)


@instr("sltiu", instruction_type=InstructionType.I.SCALAR)
def sltiu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = args.imm & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr("xori", instruction_type=InstructionType.I.SCALAR)
def xori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ args.imm)


@instr("srli", instruction_type=InstructionType.I.SCALAR)
def srli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> args.imm)


@instr("srai", instruction_type=InstructionType.I.SCALAR)
def srai(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> args.imm)


@instr("ori", instruction_type=InstructionType.I.SCALAR)
def ori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | args.imm)


@instr("andi", instruction_type=InstructionType.I.SCALAR)
def andi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & args.imm)


@instr("auipc", instruction_type=InstructionType.U.SCALAR)
def auipc(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, (args.imm << 20) & 0xFFFFFFFF + state.pc)


@instr("sb", instruction_type=InstructionType.S.SCALAR)
def sb(state: ArchState, args: ScalarArgs) -> None:
    state.write_vmem(args.rs1, args.imm, state.read_xrf(args.rs1) & 0xFF)


@instr("sh", instruction_type=InstructionType.S.SCALAR)
def sh(state: ArchState, args: ScalarArgs) -> None:
    state.write_vmem(args.rs1, args.imm, state.read_xrf(args.rs1) & 0xFFFF)


@instr("sw", instruction_type=InstructionType.S.SCALAR)
def sw(state: ArchState, args: ScalarArgs) -> None:
    state.write_vmem(args.rs1, args.imm, state.read_xrf(args.rs1) & 0xFFFFFFFF)


@instr("add", instruction_type=InstructionType.R.SCALAR)
def add(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + state.xrf[args.rs2])


@instr("sub", instruction_type=InstructionType.R.SCALAR)
def sub(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] - state.xrf[args.rs2])


@instr("sll", instruction_type=InstructionType.R.SCALAR)
def sll(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << state.xrf[args.rs2])


@instr("slt", instruction_type=InstructionType.R.SCALAR)
def slt(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < state.xrf[args.rs2] else 0)


@instr("sltu", instruction_type=InstructionType.R.SCALAR)
def sltu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr("xor", instruction_type=InstructionType.R.SCALAR)
def xor(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ state.xrf[args.rs2])


@instr("srl", instruction_type=InstructionType.R.SCALAR)
def srl(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr("sra", instruction_type=InstructionType.R.SCALAR)
def sra(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr("or", instruction_type=InstructionType.R.SCALAR)
def or_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | state.xrf[args.rs2])


@instr("and", instruction_type=InstructionType.R.SCALAR)
def and_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & state.xrf[args.rs2])


@instr("lui", instruction_type=InstructionType.U.SCALAR)
def lui(state: ArchState, args: ScalarArgs) -> None:
    """Load upper immediate: rd = imm << 12 (RISC-V LUI semantics)."""
    state.write_xrf(args.rd, (args.imm << 12) & _MASK64)


@instr("vadd.bf16", instruction_type=InstructionType.VR.VECTOR)
def vadd_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, (a + b).to(torch.bfloat16))


@instr("vredsum.bf16", instruction_type=InstructionType.VR.VECTOR)
def vredsum_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    summed = x.sum(dim=0)
    result = torch.zeros(x.shape)
    result[0:,] = summed
    state.write_mrf_bf16(result)


@instr("vsub.bf16", instruction_type=InstructionType.VR.VECTOR)
def vsub_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    state.write_mrf_bf16(args.vd, (a - b).to(torch.bfloat16))


@instr("vmin.bf16", instruction_type=InstructionType.VR.VECTOR)
def vmin_bf16(state: ArchState, args: VectorArgs) -> None:
    pass  # TODO: implementation missing in original


@instr("vmax.bf16", instruction_type=InstructionType.VR.VECTOR)
def vmax_bf16(state: ArchState, args: VectorArgs) -> None:
    pass  # TODO: implementation missing in original


@instr("vmul.bf16", instruction_type=InstructionType.VR.VECTOR)
def vmul_bf16(state: ArchState, args: VectorArgs) -> None:
    a = state.read_mrf_bf16(args.vs1)
    b = state.read_mrf_bf16(args.vs2)
    result = (a * b).to(torch.bfloat16)
    state.write_mrf_bf16(args.vd, result)


@instr("vmov", instruction_type=InstructionType.VR.VECTOR)
def vmov(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, state.read_mrf_bf16(args.vs1))


@instr("vrecip.bf16", instruction_type=InstructionType.VR.VECTOR)
def vrecip_bf16(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, 1.0 / state.read_mrf_bf16(args.vs1))


@instr("vexp.bf16", instruction_type=InstructionType.VR.VECTOR)
def vexp_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.exp(x).to(torch.bfloat16))


@instr("vexp2.bf16", instruction_type=InstructionType.VR.VECTOR)
def vexp2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.exp2(x).to(torch.bfloat16))


@instr("vpack.bf16.fp8", instruction_type=InstructionType.VR.VECTOR)
def vpack_bf16_fp8(state: ArchState, args: VectorArgs) -> None:
    assert args.vs1 != state.cfg.num_m_registers - 1

    scale = state.read_erf(args.es1)

    reg_low = state.read_mrf_bf16(args.vs1)
    reg_high = state.read_mrf_bf16(args.vs1 + 1)
    combined_bf16 = torch.cat([reg_low, reg_high], dim=1)

    quantized_fp8 = (combined_bf16 * scale).to(torch.float8_e4m3fn)

    state.write_mrf_fp8(args.vd, quantized_fp8)


@instr("vunpack.fp8.bf16", instruction_type=InstructionType.VR.VECTOR)
def vunpack_fp8_bf16(state: ArchState, args: VectorArgs) -> None:
    assert args.vd != state.cfg.num_m_registers - 1

    scale = state.read_erf(args.es1)

    source_fp8 = state.read_mrf_fp8(args.vs1)
    dequantized_bf16 = source_fp8.to(torch.bfloat16)

    scaled_bf16 = dequantized_bf16 / scale

    reg_low, reg_high = torch.chunk(scaled_bf16, chunks=2, dim=1)
    state.write_mrf_bf16(args.vd, reg_low)
    state.write_mrf_bf16(args.vd + 1, reg_high)


@instr("vrelu.bf16", instruction_type=InstructionType.VR.VECTOR)
def vrelu_bf16(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, torch.relu(state.read_mrf_bf16(args.vs1)))


@instr("vsin.bf16", instruction_type=InstructionType.VR.VECTOR)
def vsin_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.sin(x).to(torch.bfloat16))


@instr("vcos.bf16", instruction_type=InstructionType.VR.VECTOR)
def vcos_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.cos(x).to(torch.bfloat16))


@instr("vtanh.bf16", instruction_type=InstructionType.VR.VECTOR)
def vtanh_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)  # fixed missing x
    state.write_mrf_bf16(args.vd, torch.tanh(x).to(torch.bfloat16))


@instr("vlog2.bf16", instruction_type=InstructionType.VR.VECTOR)
def vlog2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.log2(x).to(torch.bfloat16))


@instr("vsqrt.bf16", instruction_type=InstructionType.VR.VECTOR)
def vsqrt_bf16(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1)
    state.write_mrf_bf16(args.vd, torch.sqrt(x).to(torch.bfloat16))


@instr("vli.all", instruction_type=InstructionType.VI.VECTOR)
def vli_all(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    state.write_mrf_bf16(args.vd, torch.full(shape, args.imm, dtype=torch.bfloat16))


@instr("vli.row", instruction_type=InstructionType.VI.VECTOR)
def vli_row(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, :] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr("vli.col", instruction_type=InstructionType.VI.VECTOR)
def vli_col(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[:, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr("vli.one", instruction_type=InstructionType.VI.VECTOR)
def vli_one(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr("beq", instruction_type=InstructionType.SB.SCALAR)
def beq(state: ArchState, args: ScalarArgs) -> None:
    if state.xrf[args.rs1] == state.xrf[args.rs2]:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bne", instruction_type=InstructionType.SB.SCALAR)
def bne(state: ArchState, args: ScalarArgs) -> None:
    if state.xrf[args.rs1] != state.xrf[args.rs2]:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("blt", instruction_type=InstructionType.SB.SCALAR)
def blt(state: ArchState, args: ScalarArgs) -> None:
    if state.xrf[args.rs1] < state.xrf[args.rs2]:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bge", instruction_type=InstructionType.SB.SCALAR)
def bge(state: ArchState, args: ScalarArgs) -> None:
    """Branch if rs1 >= rs2 (signed)."""
    if state.xrf[args.rs1] >= state.xrf[args.rs2]:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bltu", instruction_type=InstructionType.SB.SCALAR)
def bltu(state: ArchState, args: ScalarArgs) -> None:
    """Branch if rs1 < rs2 (unsigned)."""
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a < b:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("bgeu", instruction_type=InstructionType.SB.SCALAR)
def bgeu(state: ArchState, args: ScalarArgs) -> None:
    """Branch if rs1 >= rs2 (unsigned)."""
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a >= b:
        state.set_npc(
            state.pc + args.imm - PIPELINE_LATENCY
        )  # FIXME: this is a hack to compensate for the IF->EX delay


@instr("jalr", instruction_type=InstructionType.UJ.SCALAR)
def jalr(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(
        state.read_xrf(args.rs1) + args.imm - PIPELINE_LATENCY
    )  # FIXME: this is a hack to


@instr("delay", instruction_type=InstructionType.I.DELAY)
def delay(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr("vtrpose.xlu", instruction_type=InstructionType.VR.VECTOR)
def vtrpose_xlu(state: ArchState, args: VectorArgs) -> None:
    assert args.vs1 != state.cfg.num_m_registers - 1
    assert args.vd != state.cfg.num_m_registers - 1
    reg_a = state.read_mrf_bf16(args.vs1)
    reg_b = state.read_mrf_bf16(args.vs1 + 1)
    combined = torch.stack([reg_a, reg_b])
    reshaped = combined.view(2, 16, 2)
    dst_0 = reshaped[:, :, 0].t().reshape(-1)
    dst_1 = reshaped[:, :, 1].t().reshape(-1)
    state.write_mrf_bf16(args.vd, dst_0)
    state.write_mrf_bf16(args.vd + 1, dst_1)


@instr("vreduce.max.xlu", instruction_type=InstructionType.VR.VECTOR)
def vreduce_max_xlu(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1).max(dim=1).values
    state.write_mrf_bf16(args.vd, x)


@instr("vreduce.sum.xlu", instruction_type=InstructionType.VR.VECTOR)
def vreduce_sum_xlu(state: ArchState, args: VectorArgs) -> None:
    x = state.read_mrf_bf16(args.vs1).sum(dim=1)
    state.write_mrf_bf16(args.vd, x)


@instr("jal", instruction_type=InstructionType.UJ.SCALAR)
def jal(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(
        state.pc + args.imm - PIPELINE_LATENCY
    )  # FIXME: this is a hack to compensate for the IF->EX delay


# TODO: Implement 'ecall'
@instr("ecall", instruction_type=InstructionType.I.SCALAR)
def ecall(state: ArchState, args: ScalarArgs) -> None:
    pass


# TODO: Implement 'ebreak'
@instr("ebreak", instruction_type=InstructionType.I.SCALAR)
def ebreak(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr("vmatpush.weight.mxu0", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_weight_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_wb_u8("mxu0", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr("vmatpush.weight.mxu1", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_weight_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_wb_u8("mxu1", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr("vmatpush.acc.fp8.mxu0", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_acc_fp8_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu0",
        _acc_dest_index(args),
        state.read_mrf_fp8(args.vs1).to(torch.bfloat16),
    )


@instr("vmatpush.acc.fp8.mxu1", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_acc_fp8_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu1",
        _acc_dest_index(args),
        state.read_mrf_fp8(args.vs1).to(torch.bfloat16),
    )


@instr("vmatpush.acc.bf16.mxu0", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_acc_bf16_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu0", _acc_dest_index(args), state.read_mrf_bf16_tile(args.vs1)
    )


@instr("vmatpush.acc.bf16.mxu1", instruction_type=InstructionType.VR.VECTOR)
def vmatpush_acc_bf16_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_acc_bf16(
        "mxu1", _acc_dest_index(args), state.read_mrf_bf16_tile(args.vs1)
    )


@instr("vmatpop.fp8.acc.mxu0", instruction_type=InstructionType.VR.VECTOR)
def vmatpop_fp8_acc_mxu0(state: ArchState, args: VectorArgs) -> None:
    quantized = state.read_acc_bf16("mxu0", _acc_source_index(args)).to(
        torch.float8_e4m3fn
    )
    state.write_mrf_u8(args.vd, quantized.view(torch.uint8))


@instr("vmatpop.fp8.acc.mxu1", instruction_type=InstructionType.VR.VECTOR)
def vmatpop_fp8_acc_mxu1(state: ArchState, args: VectorArgs) -> None:
    quantized = state.read_acc_bf16("mxu1", _acc_source_index(args)).to(
        torch.float8_e4m3fn
    )
    state.write_mrf_u8(args.vd, quantized.view(torch.uint8))


@instr("vmatpop.bf16.acc.mxu0", instruction_type=InstructionType.VR.VECTOR)
def vmatpop_bf16_acc_mxu0(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16_tile(
        args.vd, state.read_acc_bf16("mxu0", _acc_source_index(args))
    )


@instr("vmatpop.bf16.acc.mxu1", instruction_type=InstructionType.VR.VECTOR)
def vmatpop_bf16_acc_mxu1(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16_tile(
        args.vd, state.read_acc_bf16("mxu1", _acc_source_index(args))
    )


@instr("vmatmul.mxu0", instruction_type=InstructionType.VR.MATRIX_SYSTOLIC)
def vmatmul_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=False)


@instr("vmatmul.mxu1", instruction_type=InstructionType.VR.MATRIX_IPT)
def vmatmul_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=False)


@instr("vmatmul.acc.mxu0", instruction_type=InstructionType.VR.MATRIX_SYSTOLIC)
def vmatmul_acc_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=True)


@instr("vmatmul.acc.mxu1", instruction_type=InstructionType.VR.MATRIX_IPT)
def vmatmul_acc_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=True)


@instr("dma.load.ch<N>", instruction_type=InstructionType.R.DMA)
def dma_load_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_dram(state.read_xrf(args.rs1), length)
    state.write_vmem(state.read_xrf(args.rd), 0, data)


@instr("dma.store.ch<N>", instruction_type=InstructionType.R.DMA)
def dma_store_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_vmem(state.read_xrf(args.rs1), 0, length)
    state.write_dram(state.read_xrf(args.rd), data)


@instr("dma.config.ch<N>", instruction_type=InstructionType.I.DMA)
def dma_config_ch_n(state: ArchState, args: DmaArgs) -> None:
    state.base = state.read_xrf(args.rs1)


@instr("dma.wait.ch<N>", instruction_type=InstructionType.I.BARRIER)
def dma_wait_ch_n(state: ArchState, args: DmaArgs) -> None:
    pass


# # =============================================================================
# # Extras not in README
# # =============================================================================


# @instr("mv.mw", instruction_type=InstructionType.MATRIX)
# def mv_mw(state: ArchState, args: MatrixArgs) -> None:
#     """
#     Vector/matrix move from matrix registers to weight buffer.
#     """
#     # TODO: check register dimensions
#     state.write_wb_bf16("mxu0", args.rd, state.read_mrf_bf16(args.rs1))


# @instr("vsin", instruction_type=InstructionType.VECTOR)
# def vsin(state: ArchState, args: VectorArgs) -> None:
#     x = state.read_mrf_bf16(args.vs1)
#     state.write_mrf_bf16(args.vd, torch.sin(x).to(torch.bfloat16))


# @instr("vcos", instruction_type=InstructionType.VECTOR)
# def vcos(state: ArchState, args: VectorArgs) -> None:
#     x = state.read_mrf_bf16(args.vs1)
#     state.write_mrf_bf16(args.vd, torch.cos(x).to(torch.bfloat16))


# @instr("vtanh", instruction_type=InstructionType.VECTOR)
# def vtanh(state: ArchState, args: VectorArgs) -> None:
#     x = state.read_mrf_bf16(args.vs1)  # fixed missing x
#     state.write_mrf_bf16(args.vd, torch.tanh(x).to(torch.bfloat16))


# @instr("vrot.reduce.sum", instruction_type=InstructionType.VECTOR)
# def vrot_reduce_sum(state: ArchState, args: VectorArgs) -> None:
#     """Reduce sum over last (across rows) dimension. For (rows, cols) in, gives (rows, 1) broadcast."""
#     # TODO: implementation cost?
#     x = state.read_mrf_bf16(args.vs1)
#     sum_val = torch.sum(x.float(), dim=-1, keepdim=True)
#     out = sum_val.expand_as(x).to(torch.bfloat16)
#     state.write_mrf_bf16(args.vd, out)


# @instr("mv.mm", instruction_type=InstructionType.VECTOR)
# def mv_mm(state: ArchState, args: VectorArgs) -> None:
#     """
#     Vector/matrix move between matrix registers.
#     """
#     state.write_mrf_f32(args.rd, state.read_mrf_f32(args.rs1))


# @instr("vtrpose.h", instruction_type=InstructionType.VECTOR)
# def vtrpose_h(state: ArchState, args: VectorArgs) -> None:
#     """Transpose upper half: block = x[:, 0:half], write (cols, rows) with first half rows = block.T. Use with vtrpose.l + vadd for full transpose."""
#     # TODO: check correctness
#     x = state.read_mrf_bf16(args.vs1)
#     half = x.shape[0] // 2
#     block = x[0:half, :]
#     transposed = block.T.contiguous()
#     out = torch.zeros_like(x)
#     out[0:half, :] = transposed
#     state.write_mrf_bf16(args.vd, out)


# @instr("vtrpose.l", instruction_type=InstructionType.VECTOR)
# def vtrpose_l(state: ArchState, args: VectorArgs) -> None:
#     """Transpose lower half: block = x[:, half:], write (cols, rows) with second half rows = block.T. Use with vtrpose.h + vadd for full transpose."""
#     # TODO: check correctness
#     x = state.read_mrf_bf16(args.vs1)
#     half = x.shape[0] // 2
#     block = x[half:, :]
#     transposed = block.T.contiguous()
#     out = torch.zeros_like(x)
#     out[half:, :] = transposed
#     state.write_mrf_bf16(args.vd, out)


# @instr("vmatpop.mxu0", instruction_type=InstructionType.VECTOR)
# def vmatpop_mxu0(state: ArchState, args: VectorArgs) -> None:
#     state.write_mrf_bf16_tile(
#         args.vd, state.read_acc_bf16("mxu0", _acc_source_index(args))
#     )


# @instr("vmatpop.mxu1", instruction_type=InstructionType.VECTOR)
# def vmatpop_mxu1(state: ArchState, args: VectorArgs) -> None:
#     state.write_mrf_bf16_tile(
#         args.vd, state.read_acc_bf16("mxu1", _acc_source_index(args))
#     )


# @instr("dma.load", instruction_type=InstructionType.DMA)
# def dma_load(state: ArchState, args: DmaArgs) -> None:
#     """
#     DMA load from memory to matrix registers.
#     """
#     base = args.base
#     size = args.size
#     data = state.read_memory(base, size)
#     # zero pad the data to the size of the MRF
#     if data.numel() < _tensor_register_bytes(state):
#         data = torch.nn.functional.pad(
#             data,
#             (
#                 0,
#                 _tensor_register_bytes(state) - data.numel(),
#             ),
#         )
#     state.write_mrf_u8(args.rd, data)


# @instr("dma.load.mxu0", instruction_type=InstructionType.DMA)
# def dma_load_mxu0(state: ArchState, args: DmaArgs) -> None:
#     """
#     DMA load from memory to weight buffer at MXU0.
#     """
#     base = args.base
#     size = args.size
#     data = state.read_memory(base, size).to(torch.uint8)
#     # zero pad the data to the size of the WB
#     if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
#         data = torch.nn.functional.pad(
#             data,
#             (
#                 0,
#                 state.cfg.wb_width // torch.uint8.itemsize - data.numel(),
#             ),
#         )
#     state.write_wb_u8("mxu0", args.rd, data)


# @instr("dma.load.mxu1", instruction_type=InstructionType.DMA)
# def dma_load_mxu1(state: ArchState, args: DmaArgs) -> None:
#     """
#     DMA load from memory to weight buffer at MXU1.
#     """
#     base = args.base
#     size = args.size
#     data = state.read_memory(base, size).to(torch.uint8)
#     # zero pad the data to the size of the WB
#     if data.numel() < state.cfg.wb_width // torch.uint8.itemsize:
#         data = torch.nn.functional.pad(
#             data,
#             (
#                 0,
#                 state.cfg.wb_width // torch.uint8.itemsize - data.numel(),
#             ),
#         )
#     state.write_wb_u8("mxu1", args.rd, data)


# @instr("dma.store", instruction_type=InstructionType.DMA)
# def dma_store(state: ArchState, args: DmaArgs) -> None:
#     """
#     DMA store from matrix registers to memory.
#     """
#     base = args.base
#     size = args.size
#     data = state.mrf[args.rs1].view(torch.uint8)
#     state.write_memory(base, data[:size])


# @instr("dma.wait", instruction_type=InstructionType.BARRIER)
# def dma_wait(state: ArchState, args: DmaArgs) -> None:
#     """
#     Wait for target DMA operations to complete.
#     """
#     pass
