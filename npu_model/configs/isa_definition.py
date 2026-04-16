from sympy import Matrix
import torch

from typing import Any
from npu_model.isa import (
    DmaArgs,
    MatrixArgs,
    ScalarArgs,
    VectorArgs,
    instr,
    InstructionType,
)
from npu_model.hardware.arch_state import ArchState


PIPELINE_LATENCY = 2

# Mask for 64-bit unsigned comparison (RISC-V RV64)
_MASK64 = 0xFFFFFFFFFFFFFFFF


# =============================================================================
# Helper Functions
# =============================================================================


def _sign_extend(value: int, length: int):
    """Sign-extends a value of a given bit length to the Python integer width."""
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def _int_to_le_bytes(data, length: int) -> torch.Tensor:
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Length must be 1, 2, or 4 bytes.")
    return torch.tensor([data], dtype=type_map[length]).view(torch.uint8).clone()


def _le_bytes_to_int(tensor: torch.Tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")
    raw_val = tensor.contiguous().view(type_map[length]).item()
    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return int(raw_val) & masks[length]


def _tensor_register_bytes(state: ArchState) -> int:
    return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize


def _vmatmul(state: ArchState, unit: str, args: MatrixArgs, accumulate: bool) -> None:
    activation_fp16 = state.read_mrf_fp8(args.vs1).to(torch.float16)
    weight_fp16 = state.read_wb_fp8(unit, args.vs2).to(torch.float16)
    result_fp16 = activation_fp16 @ weight_fp16
    if accumulate:
        result_fp16 = result_fp16 + state.read_acc_bf16(unit, args.vd).to(torch.float16)
    state.write_acc_bf16(unit, args.vd, result_fp16.to(torch.bfloat16))


def _assert_bf16_pair(state: ArchState, reg: int) -> None:
    assert reg < state.cfg.num_m_registers - 1


def _read_mrf_bf16_pair(state: ArchState, reg: int) -> torch.Tensor:
    _assert_bf16_pair(state, reg)
    return state.read_mrf_bf16_tile(reg)


def _write_mrf_bf16_pair(state: ArchState, reg: int, value: torch.Tensor) -> None:
    _assert_bf16_pair(state, reg)
    state.write_mrf_bf16_tile(reg, value.to(torch.bfloat16).contiguous())


@instr("lb", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b000)
def lb(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 1))
    state.write_xrf(args.rd, _sign_extend(value, 8))


@instr("lh", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b001)
def lh(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 2))
    state.write_xrf(args.rd, _sign_extend(value, 16))


@instr("lw", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b010)
def lw(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 4))
    state.write_xrf(args.rd, value)


@instr("lbu", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b100)
def lbu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 1))
    state.write_xrf(args.rd, value)


@instr("lhu", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b101)
def lhu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    value = _le_bytes_to_int(state.read_vmem(state.read_xrf(args.rs1), imm, 2))
    state.write_xrf(args.rd, value)


@instr(
    "seld", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b110
)
def seld(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_erf(
        args.rd,
        int(state.read_vmem(state.read_xrf(args.rs1), imm, 1).view(torch.uint8)),
    )


@instr(
    "seli", instruction_type=InstructionType.SCALAR.I, opcode=0b0000011, funct3=0b111
)
def seli(state: ArchState, args: ScalarArgs):
    state.write_erf(args.rd, _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "vload", instruction_type=InstructionType.VECTOR.VLS, opcode=0b0000111, funct2=0b00
)
def vload(state: ArchState, args: VectorArgs) -> None:
    addr = state.read_xrf(args.rs1) + (args.imm12 << 5)
    data = state.read_vmem(addr, 0, _tensor_register_bytes(state)).view(torch.uint8)
    state.write_mrf_u8(args.vd, data)


@instr(
    "vstore", instruction_type=InstructionType.VECTOR.VLS, opcode=0b0000111, funct2=0b01
)
def vstore(state: ArchState, args: VectorArgs) -> None:
    addr = state.read_xrf(args.rs1) + (args.imm12 << 5)
    data = state.read_mrf_fp8(args.vd).view(torch.uint8)
    state.write_vmem(addr, 0, data)


@instr(
    "fence", instruction_type=InstructionType.SCALAR.I, opcode=0b0001111, funct3=0b000
)
def fence(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "addi", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b000
)
def addi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "slli", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b001
)
def slli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << (args.imm & 0x3F))


@instr(
    "slti", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b010
)
def slti(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < imm else 0)


@instr(
    "sltiu", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b011
)
def sltiu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = _sign_extend(args.imm & 0xFFF, 12) & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr(
    "xori", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b100
)
def xori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "srli", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b101
)
def srli(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> (args.imm & 0x3F))


@instr(
    "srai", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b101
)
def srai(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> (args.imm & 0x3F))


@instr("ori", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b110)
def ori(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | _sign_extend(args.imm & 0xFFF, 12))


@instr(
    "andi", instruction_type=InstructionType.SCALAR.I, opcode=0b0010011, funct3=0b111
)
def andi(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & _sign_extend(args.imm & 0xFFF, 12))


@instr("auipc", instruction_type=InstructionType.SCALAR.U, opcode=0b0010111)
def auipc(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, ((args.imm << 12) & 0xFFFFFFFF) + state.pc)


@instr("sb", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b000)
def sb(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFF, 1),
    )


@instr("sh", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b001)
def sh(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFFFF, 2),
    )


@instr("sw", instruction_type=InstructionType.SCALAR.S, opcode=0b0100011, funct3=0b010)
def sw(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_vmem(
        state.read_xrf(args.rs1),
        imm,
        _int_to_le_bytes(state.read_xrf(args.rs2) & 0xFFFFFFFF, 4),
    )


@instr(
    "add",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0000000,
)
def add(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] + state.xrf[args.rs2])


@instr(
    "sub",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0100000,
)
def sub(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] - state.xrf[args.rs2])


@instr(
    "sll",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b001,
    funct7=0b0000000,
)
def sll(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] << state.xrf[args.rs2])


@instr(
    "slt",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b010,
    funct7=0b0000000,
)
def slt(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, 1 if state.xrf[args.rs1] < state.xrf[args.rs2] else 0)


@instr(
    "sltu",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b011,
    funct7=0b0000000,
)
def sltu(state: ArchState, args: ScalarArgs) -> None:
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    state.write_xrf(args.rd, 1 if a < b else 0)


@instr(
    "xor",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b100,
    funct7=0b0000000,
)
def xor(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] ^ state.xrf[args.rs2])


@instr(
    "srl",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0000000,
)
def srl(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr(
    "sra",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0100000,
)
def sra(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] >> state.xrf[args.rs2])


@instr(
    "or",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b110,
    funct7=0b0000000,
)
def or_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] | state.xrf[args.rs2])


@instr(
    "and",
    instruction_type=InstructionType.SCALAR.R,
    opcode=0b0110011,
    funct3=0b111,
    funct7=0b0000000,
)
def and_(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, state.xrf[args.rs1] & state.xrf[args.rs2])


@instr(
    "lui", instruction_type=InstructionType.SCALAR.U, opcode=0b0110111, funct7=0b0000000
)
def lui(state: ArchState, args: ScalarArgs) -> None:
    state.write_xrf(args.rd, (args.imm << 12) & _MASK64)


@instr(
    "vadd.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000000,
)
def vadd_bf16(state: ArchState, args: VectorArgs) -> None:
    a = _read_mrf_bf16_pair(state, args.vs1)
    b = _read_mrf_bf16_pair(state, args.vs2)
    _write_mrf_bf16_pair(state, args.vd, a + b)


@instr(
    "vredsum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000001,
)
def vredsum_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = x.sum(dim=0, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vsub.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000010,
)
def vsub_bf16(state: ArchState, args: VectorArgs) -> None:
    a = _read_mrf_bf16_pair(state, args.vs1)
    b = _read_mrf_bf16_pair(state, args.vs2)
    _write_mrf_bf16_pair(state, args.vd, a - b)


@instr(
    "vmul.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000011,
)
def vmul_bf16(state: ArchState, args: VectorArgs) -> None:
    a = _read_mrf_bf16_pair(state, args.vs1)
    b = _read_mrf_bf16_pair(state, args.vs2)
    _write_mrf_bf16_pair(state, args.vd, a * b)


@instr(
    "vminimum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000100,
)
def vminimum_bf16(state: ArchState, args: VectorArgs) -> None:
    a = _read_mrf_bf16_pair(state, args.vs1)
    b = _read_mrf_bf16_pair(state, args.vs2)
    _write_mrf_bf16_pair(state, args.vd, torch.minimum(a, b))


@instr(
    "vredmin.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000101,
)
def vredmin_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = (
        x.min(dim=0, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vmaximum.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000110,
)
def vmaximum_bf16(state: ArchState, args: VectorArgs) -> None:
    a = _read_mrf_bf16_pair(state, args.vs1)
    b = _read_mrf_bf16_pair(state, args.vs2)
    _write_mrf_bf16_pair(state, args.vd, torch.maximum(a, b))


@instr(
    "vredmax.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0000111,
)
def vredmax_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = (
        x.max(dim=0, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vredsum.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100001,
)
def vredsum_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = x.sum(dim=1, keepdim=True).to(torch.bfloat16).expand_as(x).contiguous()
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vredmin.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100100,
)
def vredmin_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = (
        x.min(dim=1, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vredmax.row.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b0100110,
)
def vredmax_row_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    result = (
        x.max(dim=1, keepdim=True).values.to(torch.bfloat16).expand_as(x).contiguous()
    )
    _write_mrf_bf16_pair(state, args.vd, result)


@instr(
    "vmov",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000000,
)
def vmov(state: ArchState, args: VectorArgs) -> None:
    state.write_mrf_bf16(args.vd, state.read_mrf_bf16(args.vs1))


@instr(
    "vrecip.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000001,
)
def vrecip_bf16(state: ArchState, args: VectorArgs) -> None:
    _write_mrf_bf16_pair(state, args.vd, 1.0 / _read_mrf_bf16_pair(state, args.vs1))


@instr(
    "vexp.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000010,
)
def vexp_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.exp(x))


@instr(
    "vexp2.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000011,
)
def vexp2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.exp2(x))


@instr(
    "vpack.bf16.fp8",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000100,
)
def vpack_bf16_fp8(state: ArchState, args: VectorArgs) -> None:
    assert args.vs1 != state.cfg.num_m_registers - 1
    scale = state.read_erf(args.es1)
    reg_low = state.read_mrf_bf16(args.vs1)
    reg_high = state.read_mrf_bf16(args.vs1 + 1)
    combined_bf16 = torch.cat([reg_low, reg_high], dim=1)
    quantized_fp8 = (combined_bf16 * scale).to(torch.float8_e4m3fn)
    state.write_mrf_fp8(args.vd, quantized_fp8)


@instr(
    "vunpack.fp8.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1000101,
)
def vunpack_fp8_bf16(state: ArchState, args: VectorArgs) -> None:
    assert args.vd != state.cfg.num_m_registers - 1
    scale = state.read_erf(args.es1)
    source_fp8 = state.read_mrf_fp8(args.vs1)
    dequantized_bf16 = source_fp8.to(torch.bfloat16)
    scaled_bf16 = dequantized_bf16 / scale
    reg_low, reg_high = torch.chunk(scaled_bf16, chunks=2, dim=1)
    state.write_mrf_bf16(args.vd, reg_low)
    state.write_mrf_bf16(args.vd + 1, reg_high)


@instr(
    "vrelu.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001000,
)
def vrelu_bf16(state: ArchState, args: VectorArgs) -> None:
    _write_mrf_bf16_pair(state, args.vd, torch.relu(_read_mrf_bf16_pair(state, args.vs1)))


@instr(
    "vsin.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001001,
)
def vsin_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.sin(x))


@instr(
    "vcos.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001010,
)
def vcos_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.cos(x))


@instr(
    "vtanh.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001011,
)
def vtanh_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.tanh(x))


@instr(
    "vlog2.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001100,
)
def vlog2_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.log2(x))


@instr(
    "vsqrt.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001101,
)
def vsqrt_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, torch.sqrt(x))


@instr(
    "vsquare.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001110,
)
def vsquare_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, x * x)


@instr(
    "vcube.bf16",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1010111,
    funct7=0b1001111,
)
def vcube_bf16(state: ArchState, args: VectorArgs) -> None:
    x = _read_mrf_bf16_pair(state, args.vs1)
    _write_mrf_bf16_pair(state, args.vd, x * x * x)


@instr(
    "vli.all",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b000,
)
def vli_all(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    state.write_mrf_bf16(args.vd, torch.full(shape, args.imm, dtype=torch.bfloat16))


@instr(
    "vli.row",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b001,
)
def vli_row(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, :] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "vli.col",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b010,
)
def vli_col(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[:, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "vli.one",
    instruction_type=InstructionType.VECTOR.VI,
    opcode=0b1011111,
    funct3=0b011,
)
def vli_one(state: ArchState, args: VectorArgs) -> None:
    shape = state.read_mrf_bf16(0).shape
    x = torch.zeros(shape, dtype=torch.bfloat16)
    x[0, 0] = args.imm
    state.write_mrf_bf16(args.vd, x)


@instr(
    "beq", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b000
)
def beq(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] == state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bne", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b001
)
def bne(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] != state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "blt", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b100
)
def blt(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] < state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bge", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b101
)
def bge(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    if state.xrf[args.rs1] >= state.xrf[args.rs2]:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bltu", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b110
)
def bltu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a < b:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "bgeu", instruction_type=InstructionType.SCALAR.SB, opcode=0b1100011, funct3=0b111
)
def bgeu(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0x1FFF, 13)
    a = state.xrf[args.rs1] & _MASK64
    b = state.xrf[args.rs2] & _MASK64
    if a >= b:
        state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "jalr", instruction_type=InstructionType.SCALAR.I, opcode=0b1100111, funct3=0b000
)
def jalr(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFF, 12)
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(state.read_xrf(args.rs1) + imm - PIPELINE_LATENCY)


@instr(
    "delay", instruction_type=InstructionType.DELAY.I, opcode=0b1100111, funct3=0b001
)
def delay(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "vtrpose.xlu",
    instruction_type=InstructionType.VECTOR.VR,
    opcode=0b1101011,
    funct7=0b0000000,
)
def vtrpose_xlu(state: ArchState, args: VectorArgs) -> None:
    reg_in = state.read_mrf_fp8(args.vs1)
    transposed = reg_in.view(32, 32).t().contiguous().reshape(-1)
    state.write_mrf_fp8(args.vd, transposed)


@instr("jal", instruction_type=InstructionType.SCALAR.UJ, opcode=0b1101111)
def jal(state: ArchState, args: ScalarArgs) -> None:
    imm = _sign_extend(args.imm & 0xFFFFF, 20)
    state.write_xrf(args.rd, state.pc + 4)
    state.set_npc(state.pc + imm - PIPELINE_LATENCY)


@instr(
    "csrrw", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b001
)
def csrrw(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrs", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b010
)
def csrrs(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old | state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrc", instruction_type=InstructionType.SCALAR.CSR, opcode=0b1110011, funct3=0b011
)
def csrrc(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old & ~state.read_xrf(args.rs1))
    state.write_xrf(args.rd, old)


@instr(
    "csrrwi",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b101,
)
def csrrwi(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, args.rs1 & 0b11111)
    state.write_xrf(args.rd, old)


@instr(
    "csrrsi",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b110,
)
def csrrsi(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old | (args.rs1 & 0b11111))
    state.write_xrf(args.rd, old)


@instr(
    "csrrci",
    instruction_type=InstructionType.SCALAR.CSR,
    opcode=0b1110011,
    funct3=0b100,
)
def csrrci(state: ArchState, args: ScalarArgs) -> None:
    old = state.read_csrf(args.imm)
    state.write_csrf(args.imm, old & ~(args.rs1 & 0b11111))
    state.write_xrf(args.rd, old)


@instr(
    "ecall", instruction_type=InstructionType.SCALAR.I, opcode=0b1110011, funct3=0b000
)
def ecall(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "ebreak", instruction_type=InstructionType.SCALAR.I, opcode=0b1110011, funct3=0b000
)
def ebreak(state: ArchState, args: ScalarArgs) -> None:
    pass


@instr(
    "vmatpush.weight.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0000000,
)
def vmatpush_weight_mxu0(state: ArchState, args: MatrixArgs) -> None:
    state.write_wb_u8("mxu0", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr(
    "vmatpush.weight.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0000001,
)
def vmatpush_weight_mxu1(state: ArchState, args: MatrixArgs) -> None:
    state.write_wb_u8("mxu1", args.vd, state.mrf[args.vs1].view(torch.uint8))


@instr(
    "vmatpush.acc.fp8.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0000010,
)
def vmatpush_acc_fp8_mxu0(state: ArchState, args: MatrixArgs) -> None:
    state.write_acc_bf16(
        "mxu0", args.vd, state.read_mrf_fp8(args.vs1).to(torch.bfloat16)
    )


@instr(
    "vmatpush.acc.fp8.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0000011,
)
def vmatpush_acc_fp8_mxu1(state: ArchState, args: MatrixArgs) -> None:
    state.write_acc_bf16(
        "mxu1", args.vd, state.read_mrf_fp8(args.vs1).to(torch.bfloat16)
    )


@instr(
    "vmatpush.acc.bf16.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0000100,
)
def vmatpush_acc_bf16_mxu0(state: ArchState, args: MatrixArgs) -> None:
    state.write_acc_bf16("mxu0", args.vd, state.read_mrf_bf16_tile(args.vs1))


@instr(
    "vmatpush.acc.bf16.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0000101,
)
def vmatpush_acc_bf16_mxu1(state: ArchState, args: MatrixArgs) -> None:
    state.write_acc_bf16("mxu1", args.vd, state.read_mrf_bf16_tile(args.vs1))


@instr(
    "vmatpop.fp8.acc.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0000110,
)
def vmatpop_fp8_acc_mxu0(state: ArchState, args: MatrixArgs) -> None:
    quantized = state.read_acc_bf16("mxu0", args.vs1).to(torch.float8_e4m3fn)
    state.write_mrf_u8(args.vd, quantized.view(torch.uint8))


@instr(
    "vmatpop.fp8.acc.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0000111,
)
def vmatpop_fp8_acc_mxu1(state: ArchState, args: MatrixArgs) -> None:
    quantized = state.read_acc_bf16("mxu1", args.vs1).to(torch.float8_e4m3fn)
    state.write_mrf_fp8(args.vd, quantized.view(torch.uint8))


@instr(
    "vmatpop.bf16.acc.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0001000,
)
def vmatpop_bf16_acc_mxu0(state: ArchState, args: MatrixArgs) -> None:
    state.write_mrf_bf16_tile(args.vd, state.read_acc_bf16("mxu0", args.vs1))


@instr(
    "vmatpop.bf16.acc.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0001001,
)
def vmatpop_bf16_acc_mxu1(state: ArchState, args: MatrixArgs) -> None:
    state.write_mrf_bf16_tile(args.vd, state.read_acc_bf16("mxu1", args.vs1))


@instr(
    "vmatmul.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0001010,
)
def vmatmul_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=False)


@instr(
    "vmatmul.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0001011,
)
def vmatmul_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=False)


@instr(
    "vmatmul.acc.mxu0",
    instruction_type=InstructionType.MATRIX_SYSTOLIC.VR,
    opcode=0b1110111,
    funct7=0b0001100,
)
def vmatmul_acc_mxu0(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu0", args, accumulate=True)


@instr(
    "vmatmul.acc.mxu1",
    instruction_type=InstructionType.MATRIX_IPT.VR,
    opcode=0b1110111,
    funct7=0b0001101,
)
def vmatmul_acc_mxu1(state: ArchState, args: MatrixArgs) -> None:
    _vmatmul(state, "mxu1", args, accumulate=True)


@instr(
    "dma.load.ch<N>",
    instruction_type=InstructionType.DMA.R,
    opcode=0b1111011,
    funct7=0b0000000,
)
def dma_load_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_dram(state.read_xrf(args.rs1), length)
    state.write_vmem(state.read_xrf(args.rd), 0, data)


@instr(
    "dma.store.ch<N>",
    instruction_type=InstructionType.DMA.R,
    opcode=0b1111011,
    funct7=0b0000001,
)
def dma_store_ch_n(state: ArchState, args: DmaArgs) -> None:
    length = state.read_xrf(args.rs2)
    data = state.read_vmem(state.read_xrf(args.rs1), 0, length)
    state.write_dram(state.read_xrf(args.rd), data)


@instr(
    "dma.config.ch<N>",
    instruction_type=InstructionType.DMA.I,
    opcode=0b1111111,
    funct7=0b0000000,
)
def dma_config_ch_n(state: ArchState, args: DmaArgs) -> None:
    state.base = state.read_xrf(args.rs1)


@instr(
    "dma.wait.ch<N>",
    instruction_type=InstructionType.BARRIER.I,
    opcode=0b1111111,
    funct7=0b0000001,
)
def dma_wait_ch_n(state: ArchState, args: DmaArgs) -> None:
    pass
