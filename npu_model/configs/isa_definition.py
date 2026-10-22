from typing import TYPE_CHECKING
import torch
from npu_model.isa import (
    CSRType,
    IType,
    RType,
    SBType,
    SType,
    UJType,
    UType,
    VIType,
    VLSType,
    VRType,
)
from npu_model.isa_patterns import (
    DirectImm,
    CSRImm,
    CSRReg,
    DMAConfig,
    DMATransfer,
    DMAWait,
    ExponentImm,
    ExponentOffsetLoad,
    Nullary,
    ScalarBaseOffsetStore,
    ScalarComputeImm,
    ScalarComputeShamt,
    ScalarBranchImm,
    ScalarComputeReg,
    ScalarImm,
    ScalarOffsetLoad,
    JalrPattern,
    TensorBaseOffset,
    TensorComputeBinary,
    TensorComputeMixed,
    TensorComputeUnary,
    MXUWeightPush,
    MXUAccumulatorPush,
    MXUAccumulatorPopE1,
    MXUAccumulatorPop,
    MXUMatMul,
    UnaryImm,
)
from npu_model.isa_types import (
    ScalarReg,
    ExponentReg,
    MatrixReg,
    WeightBuffer,
    Accumulator,
    SBImm12,
    Imm12,
    EXU,
)

if TYPE_CHECKING:
    from npu_model.hardware.arch_state import ArchState

# ScalarCore is RV32 with word-indexed PCs.
_MASK32 = 0xFFFFFFFF


# =============================================================================
# Helper Functions
# =============================================================================


def _sign_extend(value: int, length: int):
    """Sign-extends a value of a given bit length to the Python integer width."""
    value &= (1 << length) - 1
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def _int_to_le_bytes(data: int, length: int) -> torch.Tensor:
    if length not in (1, 2, 4):
        raise ValueError("Length must be 1, 2, or 4 bytes.")
    raw = (data & ((1 << (8 * length)) - 1)).to_bytes(length, "little")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _le_bytes_to_int(tensor: torch.Tensor) -> int:
    length = tensor.numel()
    type_map = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
    if length not in type_map:
        raise ValueError("Tensor length must be 1, 2, or 4 bytes.")
    raw_val = tensor.contiguous().view(type_map[length]).item()
    masks = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}
    return int(raw_val) & masks[length]


def _vmem_address(state: ArchState, rs1: int, imm: int) -> int:
    """Scalar LSU byte address; only the VMEM-local address bits reach the LSU."""
    mask = (1 << (state.cfg.vmem_size - 1).bit_length()) - 1
    return (state.read_xrf(rs1) + _sign_extend(imm & 0xFFF, 12)) & mask


def _vmem_line_address(state: ArchState, rs1: int, imm: int) -> int:
    """VLS byte address. Operands count words; ScalarCore drops the three
    word-offset bits and the LSU expands the 32-byte line address."""
    lines = (state.cfg.vmem_size // 32 - 1).bit_length()
    words = state.read_xrf(rs1) + (_sign_extend(imm & 0xFFF, 12) << 5)
    return ((words >> 3) & ((1 << lines) - 1)) * 32


def _load_word(state: ArchState, rs1: int, imm: int) -> tuple[int, int]:
    """Scalar loads fetch the aligned word holding the byte address."""
    address = _vmem_address(state, rs1, imm)
    return address, _le_bytes_to_int(state.read_vmem(address & ~3, 0, 4))


def _store(state: ArchState, rs1: int, rs2: int, imm: int, size: int) -> None:
    # Hardware ignores bit 0 for halfwords and bits 1:0 for words.
    address = _vmem_address(state, rs1, imm) & ~(size - 1)
    state.write_vmem(address, 0, _int_to_le_bytes(state.read_xrf(rs2), size))


def _tensor_register_bytes(state: ArchState) -> int:
    return state.cfg.mrf_depth * state.cfg.mrf_width // torch.uint8.itemsize


def _vmatmul(
    state: ArchState,
    unit: str,
    vd: Accumulator,
    vs1: MatrixReg,
    vs2: WeightBuffer,
    accumulate: bool,
) -> None:
    # Weight-buffer row j holds output column j's weights: acc = A @ W^T (+ acc).
    a = state.read_mrf_fp8(vs1)
    b = state.read_wb_fp8(unit, vs2).T
    if accumulate:
        c = state.read_acc_bf16(unit, vd)
    else:
        c = torch.zeros(a.shape[0], b.shape[1], dtype=torch.bfloat16)
    # mxu0 is the systolic array, mxu1 the inner-product trees.
    matmul = state.math.systolic_matmul if unit == "mxu0" else state.math.inner_product_matmul
    state.write_acc_bf16(unit, vd, matmul(a, b, c))


def _assert_bf16_pair(state: ArchState, reg: int) -> None:
    assert reg < state.cfg.num_m_registers - 1


def _read_mrf_bf16_pair(state: ArchState, reg: int) -> torch.Tensor:
    _assert_bf16_pair(state, reg)
    return state.read_mrf_bf16_tile(reg)


def _write_mrf_bf16_pair(state: ArchState, reg: int, value: torch.Tensor) -> None:
    _assert_bf16_pair(state, reg)
    state.write_mrf_bf16_tile(reg, value.to(torch.bfloat16).contiguous())


def _bf16_register_shape(state: ArchState) -> tuple[int, int]:
    return state.cfg.mrf_depth, state.cfg.mrf_width // torch.bfloat16.itemsize


def _bf16_pair_shape(state: ArchState) -> tuple[int, int]:
    """A pair is one tile: m[reg] holds the left columns, m[reg + 1] the right."""
    rows, columns = _bf16_register_shape(state)
    return rows, 2 * columns


def _read_mrf_bf16_rows(state: ArchState, reg: int) -> torch.Tensor:
    """A pair in register order: m[reg]'s rows, then m[reg + 1]'s."""
    _assert_bf16_pair(state, reg)
    return torch.cat((state.read_mrf_bf16(reg), state.read_mrf_bf16(reg + 1)))


def _write_mrf_bf16_rows(state: ArchState, reg: int, value: torch.Tensor) -> None:
    _assert_bf16_pair(state, reg)
    state.write_mrf_bf16(reg, value[:state.cfg.mrf_depth].contiguous())
    state.write_mrf_bf16(reg + 1, value[state.cfg.mrf_depth:].contiguous())


class LB(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        address, word = _load_word(state, self.rs1, self.imm)
        state.write_xrf(self.rd, _sign_extend(word >> (8 * (address & 3)), 8))


class LH(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        address, word = _load_word(state, self.rs1, self.imm)
        state.write_xrf(self.rd, _sign_extend(word >> (8 * (address & 2)), 16))


class LW(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        _, word = _load_word(state, self.rs1, self.imm)
        state.write_xrf(self.rd, word)


class LBU(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        address, word = _load_word(state, self.rs1, self.imm)
        state.write_xrf(self.rd, (word >> (8 * (address & 3))) & 0xFF)


class LHU(ScalarOffsetLoad, IType, exu=EXU.LSU, opcode=0b0000011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        address, word = _load_word(state, self.rs1, self.imm)
        state.write_xrf(self.rd, (word >> (8 * (address & 2))) & 0xFFFF)


class SELD(
    ExponentOffsetLoad, IType[ExponentReg], exu=EXU.LSU, opcode=0b0000011, funct3=0b110
):
    def exec(self, state: ArchState) -> None:
        # ScalarCore selects memWord[7:0], without byte shifting.
        _, word = _load_word(state, self.rs1, self.imm)
        state.write_erf(self.rd, word & 0xFF)


class SELI(
    ExponentImm, IType[ExponentReg], exu=EXU.SCALAR, opcode=0b0000011, funct3=0b111
):
    def exec(self, state: ArchState):
        state.write_erf(self.rd, _sign_extend(self.imm & 0xFFF, 12))


class VLOAD(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b00):
    def exec(self, state: ArchState) -> None:
        address = _vmem_line_address(state, self.rs1, self.imm)
        state.write_mrf_u8(self.vd, state.read_vmem(address, 0, _tensor_register_bytes(state)))


class VSTORE(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b01):
    def exec(self, state: ArchState) -> None:
        address = _vmem_line_address(state, self.rs1, self.imm)
        state.write_vmem(address, 0, state.read_mrf_u8(self.vd))


class FENCE(Nullary, IType, exu=EXU.SCALAR, opcode=0b0001111, funct3=0b000):
    imm: Imm12 = Imm12(0)
    def exec(self, state: ArchState) -> None:
        pass


class ADDI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] + _sign_extend(self.imm & 0xFFF, 12)
        )


class SLLI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] << (self.imm & 0x1F))


class SLTI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0xFFF, 12)
        state.write_xrf(self.rd, 1 if _sign_extend(state.xrf[self.rs1], 32) < imm else 0)


class SLTIU(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        a = state.xrf[self.rs1] & _MASK32
        b = _sign_extend(self.imm & 0xFFF, 12) & _MASK32
        state.write_xrf(self.rd, 1 if a < b else 0)


class XORI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] ^ _sign_extend(self.imm & 0xFFF, 12)
        )


class SRLI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] >> (self.imm & 0x1F))


class SRAI(ScalarComputeShamt, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b101):
    UPPER_IMM = 0b0100000

    def exec(self, state: ArchState) -> None:
        src = _sign_extend(state.xrf[self.rs1] & 0xFFFFFFFF, 32)
        state.write_xrf(self.rd, (src >> (self.imm & 0x1F)) & 0xFFFFFFFF)


class ORI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] | _sign_extend(self.imm & 0xFFF, 12)
        )


class ANDI(ScalarComputeImm, IType, exu=EXU.SCALAR, opcode=0b0010011, funct3=0b111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, state.xrf[self.rs1] & _sign_extend(self.imm & 0xFFF, 12)
        )


class AUIPC(ScalarImm, UType, exu=EXU.SCALAR, opcode=0b0010111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(
            self.rd, ((self.imm << 12) & 0xFFFFFFFF) + state.execute_pc
        )


class SB(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        _store(state, self.rs1, self.rs2, self.imm, 1)


class SH(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        _store(state, self.rs1, self.rs2, self.imm, 2)


class SW(ScalarBaseOffsetStore, SType, exu=EXU.LSU, opcode=0b0100011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        _store(state, self.rs1, self.rs2, self.imm, 4)


class ADD(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] + state.xrf[self.rs2])


class SUB(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b000,
    funct7=0b0100000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] - state.xrf[self.rs2])


class SLL(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b001,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] << (state.xrf[self.rs2] & 0x1F))


class SLT(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b010,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, 1 if _sign_extend(state.xrf[self.rs1], 32) < _sign_extend(state.xrf[self.rs2], 32) else 0)


class SLTU(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b011,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        a = state.xrf[self.rs1] & _MASK32
        b = state.xrf[self.rs2] & _MASK32
        state.write_xrf(self.rd, 1 if a < b else 0)


class XOR(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b100,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] ^ state.xrf[self.rs2])


class SRL(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] >> (state.xrf[self.rs2] & 0x1F))


class SRA(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b101,
    funct7=0b0100000,
):
    def exec(self, state: ArchState) -> None:
        src = _sign_extend(state.xrf[self.rs1] & 0xFFFFFFFF, 32)
        state.write_xrf(self.rd, (src >> (state.xrf[self.rs2] & 0x1F)) & 0xFFFFFFFF)


class OR(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b110,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] | state.xrf[self.rs2])


class AND(
    ScalarComputeReg,
    RType,
    exu=EXU.SCALAR,
    opcode=0b0110011,
    funct3=0b111,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, state.xrf[self.rs1] & state.xrf[self.rs2])


class LUI(ScalarImm, UType, exu=EXU.SCALAR, opcode=0b0110111):
    def exec(self, state: ArchState) -> None:
        state.write_xrf(self.rd, (self.imm << 12) & _MASK32)


class VADD_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000000
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, state.math.add(a, b))


class VREDSUM_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000001
):
    def exec(self, state: ArchState) -> None:
        # Each of the 16 lanes reduces all 64 rows of the pair (both halves of
        # the tile); every row of both registers receives the 16 results.
        x = _read_mrf_bf16_rows(state, self.vs1)
        _write_mrf_bf16_rows(state, self.vd, state.math.column_sum(x).expand_as(x).contiguous())


class VSUB_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000010
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, state.math.sub(a, b))


class VMUL_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000011
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, state.math.mul(a, b))


class VMINIMUM_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000100
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, state.math.minimum(a, b))


class VREDMIN_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000101
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_rows(state, self.vs1)
        _write_mrf_bf16_rows(state, self.vd, state.math.amin(x, 0).expand_as(x).contiguous())


class VMAXIMUM_BF16(
    TensorComputeBinary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000110
):
    def exec(self, state: ArchState) -> None:
        a = _read_mrf_bf16_pair(state, self.vs1)
        b = _read_mrf_bf16_pair(state, self.vs2)
        _write_mrf_bf16_pair(state, self.vd, state.math.maximum(a, b))


class VREDMAX_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0000111
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_rows(state, self.vs1)
        _write_mrf_bf16_rows(state, self.vd, state.math.amax(x, 0).expand_as(x).contiguous())


class VREDSUM_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.row_sum(x).expand_as(x).contiguous())


class VREDMIN_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100100
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.amin(x, 1).expand_as(x).contiguous())


class VREDMAX_ROW_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b0100110
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.amax(x, 1).expand_as(x).contiguous())


class VMOV(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000000
):
    def exec(self, state: ArchState) -> None:
        _write_mrf_bf16_pair(state, self.vd, _read_mrf_bf16_pair(state, self.vs1))


class VRECIP_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("rcp", x))


class VEXP_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000010
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("exp", x))


class VEXP2_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000011
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("exp2", x))


class VPACK_BF16_FP8(
    TensorComputeMixed, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000100
):
    def exec(self, state: ArchState) -> None:
        # Converts the pair in register order (FP8 row k holds BF16 rows 2k and
        # 2k + 1), dividing by the ERF scale.
        x = _read_mrf_bf16_rows(state, self.vs2)
        state.write_mrf_fp8(self.vd, state.math.to_fp8(x.flatten(), state.read_erf(self.es1)))


class VUNPACK_FP8_BF16(
    TensorComputeMixed, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000101
):
    def exec(self, state: ArchState) -> None:
        # Inverse layout of vpack, multiplying by the ERF scale.
        x = state.read_mrf_fp8(self.vs2).flatten()
        y = state.math.from_fp8(x, state.read_erf(self.es1))
        _write_mrf_bf16_rows(state, self.vd, y.reshape(2 * state.cfg.mrf_depth, -1))


class VRELU_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001000
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("relu", x))


class VSIN_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001001
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("sin", x))


class VCOS_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001010
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("cos", x))


class VTANH_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001011
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("tanh", x))


class VLOG2_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001100
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("log2", x))


class VSQRT_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1001101
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("sqrt", x))


class VSQUARE_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000110  # Instructions.scala 57/46
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("square", x))


class VCUBE_BF16(
    TensorComputeUnary, VRType, exu=EXU.VECTOR, opcode=0b1010111, funct7=0b1000111  # Instructions.scala 57/47
):
    def exec(self, state: ArchState) -> None:
        x = _read_mrf_bf16_pair(state, self.vs1)
        _write_mrf_bf16_pair(state, self.vd, state.math.unary("cube", x))


class VLI_ALL(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        # The immediate is raw BF16 bits; vli.all and vli.row write the pair.
        bits = torch.full(_bf16_pair_shape(state), self.imm & 0xFFFF, dtype=torch.uint16)
        _write_mrf_bf16_pair(state, self.vd, bits.view(torch.bfloat16))


class VLI_ROW(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        bits = torch.zeros(_bf16_pair_shape(state), dtype=torch.uint16)
        bits[0, :] = self.imm & 0xFFFF
        _write_mrf_bf16_pair(state, self.vd, bits.view(torch.bfloat16))


class VLI_COL(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        # vli.col and vli.one write m[vd] only.
        bits = torch.zeros(_bf16_register_shape(state), dtype=torch.uint16)
        bits[:, 0] = self.imm & 0xFFFF
        state.write_mrf_bf16(self.vd, bits.view(torch.bfloat16))


class VLI_ONE(DirectImm, VIType, exu=EXU.VECTOR, opcode=0b1011111, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        bits = torch.zeros(_bf16_register_shape(state), dtype=torch.uint16)
        bits[0, 0] = self.imm & 0xFFFF
        state.write_mrf_bf16(self.vd, bits.view(torch.bfloat16))


class BEQ(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b000):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] == state.xrf[self.rs2]:
            state.set_npc(state.execute_pc + (imm >> 1))


class BNE(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if state.xrf[self.rs1] != state.xrf[self.rs2]:
            state.set_npc(state.execute_pc + (imm >> 1))


class BLT(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b100):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if _sign_extend(state.xrf[self.rs1], 32) < _sign_extend(state.xrf[self.rs2], 32):
            state.set_npc(state.execute_pc + (imm >> 1))


class BGE(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        if _sign_extend(state.xrf[self.rs1], 32) >= _sign_extend(state.xrf[self.rs2], 32):
            state.set_npc(state.execute_pc + (imm >> 1))


class BLTU(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        a = state.xrf[self.rs1] & _MASK32
        b = state.xrf[self.rs2] & _MASK32
        if a < b:
            state.set_npc(state.execute_pc + (imm >> 1))


class BGEU(ScalarBranchImm, SBType, exu=EXU.SCALAR, opcode=0b1100011, funct3=0b111):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm & 0x1FFF, 13)
        a = state.xrf[self.rs1] & _MASK32
        b = state.xrf[self.rs2] & _MASK32
        if a >= b:
            state.set_npc(state.execute_pc + (imm >> 1))


class JALR(
    JalrPattern,
    IType[ScalarReg, Imm12],
    exu=EXU.SCALAR,
    opcode=0b1100111,
    funct3=0b000,
):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm, 12)
        target = state.read_xrf(self.rs1) + imm
        state.write_xrf(self.rd, state.execute_pc + 1)
        state.set_npc(target)


class DELAY(UnaryImm, IType, exu=EXU.SCALAR, opcode=0b1100111, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        pass


class VTRPOSE_XLU(
    TensorComputeUnary, VRType, exu=EXU.XLU, opcode=0b1101011, funct7=0b0000000
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_u8(self.vd, state.read_mrf_u8(self.vs1).t().contiguous())


class JAL(ScalarImm, UJType, exu=EXU.SCALAR, opcode=0b1101111):
    def exec(self, state: ArchState) -> None:
        imm = _sign_extend(self.imm, 21)
        state.write_xrf(self.rd, state.execute_pc + 1)
        state.set_npc(state.execute_pc + (imm >> 1))


class CSRRW(CSRReg, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b001):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRS(CSRReg, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b010):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old | state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRC(CSRReg, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b011):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old & ~state.read_xrf(self.rs1))
        state.write_xrf(self.rd, old)


class CSRRWI(CSRImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b101):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, self.rs1 & 0b11111)
        state.write_xrf(self.rd, old)


class CSRRSI(CSRImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b110):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old | (self.rs1 & 0b11111))
        state.write_xrf(self.rd, old)


class CSRRCI(CSRImm, CSRType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b111):
    def exec(self, state: ArchState) -> None:
        old = state.read_csrf(self.imm)
        state.write_csrf(self.imm, old & ~(self.rs1 & 0b11111))
        state.write_xrf(self.rd, old)


class ECALL(Nullary, IType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b000):
    imm: Imm12 = Imm12(0)

    def exec(self, state: ArchState) -> None:
        state.halted = True
        state.halt_reason = "ecall"


class EBREAK(Nullary, IType, exu=EXU.SCALAR, opcode=0b1110011, funct3=0b000):
    imm: Imm12 = Imm12(1)

    def exec(self, state: ArchState) -> None:
        state.halted = True
        state.halt_reason = "ebreak"


class VMATPUSH_WEIGHT_MXU0(
    MXUWeightPush,
    VRType[WeightBuffer, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000000,
):
    def exec(self, state: ArchState) -> None:
        state.write_wb_u8("mxu0", self.vd, state.mrf[self.vs1].view(torch.uint8))


class VMATPUSH_WEIGHT_MXU1(
    MXUWeightPush,
    VRType[WeightBuffer, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000001,
):
    def exec(self, state: ArchState) -> None:
        state.write_wb_u8("mxu1", self.vd, state.mrf[self.vs1].view(torch.uint8))


class VMATPUSH_ACC_FP8_MXU0(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000010,
):
    def exec(self, state: ArchState) -> None:
        # FP8Unpack at unit scale (2**0).
        x = state.read_mrf_fp8(self.vs1)
        state.write_acc_bf16("mxu0", self.vd, state.math.from_fp8(x, 127))


class VMATPUSH_ACC_FP8_MXU1(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000011,
):
    def exec(self, state: ArchState) -> None:
        # FP8Unpack at unit scale (2**0).
        x = state.read_mrf_fp8(self.vs1)
        state.write_acc_bf16("mxu1", self.vd, state.math.from_fp8(x, 127))


class VMATPUSH_ACC_BF16_MXU0(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000100,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16("mxu0", self.vd, state.read_mrf_bf16_tile(self.vs1))


class VMATPUSH_ACC_BF16_MXU1(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000101,
):
    def exec(self, state: ArchState) -> None:
        state.write_acc_bf16("mxu1", self.vd, state.read_mrf_bf16_tile(self.vs1))


class VMATPOP_FP8_ACC_MXU0(
    MXUAccumulatorPopE1,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0000110,
):
    def exec(self, state: ArchState) -> None:
        # Multiplies by the ERF scale, unlike vpack.
        acc = state.read_acc_bf16("mxu0", self.vs2)
        fp8 = state.math.acc_to_fp8(acc, state.read_erf(self.es1))
        state.write_mrf_u8(self.vd, fp8.view(torch.uint8))


class VMATPOP_FP8_ACC_MXU1(
    MXUAccumulatorPopE1,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0000111,
):
    def exec(self, state: ArchState) -> None:
        # Multiplies by the ERF scale, unlike vpack.
        acc = state.read_acc_bf16("mxu1", self.vs2)
        fp8 = state.math.acc_to_fp8(acc, state.read_erf(self.es1))
        state.write_mrf_u8(self.vd, fp8.view(torch.uint8))


class VMATPOP_BF16_ACC_MXU0(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001000,
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_bf16_tile(self.vd, state.read_acc_bf16("mxu0", self.vs2))


class VMATPOP_BF16_ACC_MXU1(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001001,
):
    def exec(self, state: ArchState) -> None:
        state.write_mrf_bf16_tile(self.vd, state.read_acc_bf16("mxu1", self.vs2))


class VMATMUL_MXU0(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001010,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu0", self.vd, self.vs1, self.vs2, accumulate=False)


class VMATMUL_MXU1(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001011,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu1", self.vd, self.vs1, self.vs2, accumulate=False)


class VMATMUL_ACC_MXU0(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_SYSTOLIC,
    opcode=0b1110111,
    funct7=0b0001100,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu0", self.vd, self.vs1, self.vs2, accumulate=True)


class VMATMUL_ACC_MXU1(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_INNER,
    opcode=0b1110111,
    funct7=0b0001101,
):
    def exec(self, state: ArchState) -> None:
        _vmatmul(state, "mxu1", self.vd, self.vs1, self.vs2, accumulate=True)



# ── NVFP4 MXU (configs with the NVFP4 MXU; see npu_model/configs/nvfp4.py) ─────────────────────

def _nv_rows(t: torch.Tensor, width: int) -> list[list[int]]:
    return [[int(x) for x in t[i * width:(i + 1) * width]] for i in range(t.numel() // width)]


def _nv_mreg_words(state: ArchState, reg: int) -> list[list[int]]:
    """32 rows x 8 little-endian 32-bit words of m[reg]."""
    raw = _nv_rows(state.mrf[reg].view(torch.uint8), 32)
    return [[int.from_bytes(bytes(row[4 * k:4 * k + 4]), "little") for k in range(8)] for row in raw]


def _nv_write_mreg_words(state: ArchState, reg: int, rows: list[list[int]]) -> None:
    data = bytearray()
    for row in rows:
        for word in row:
            data += (word & 0xFFFF_FFFF).to_bytes(4, "little")
    state.write_mrf_u8(reg, torch.tensor(list(data), dtype=torch.uint8))


def _nv_matmul(state: ArchState, vd: int, vs1: int, ws: int, accumulate: bool) -> None:
    from npu_model.configs import nvfp4
    a = _nv_rows(state.mrf[vs1].view(torch.uint8), 32)
    sa = _nv_rows(state.mrf_scale[vs1], 4)
    w = _nv_rows(state.wb_nv[ws], 32)
    sw = _nv_rows(state.wb_nv_scale[ws], 4)
    rows = len(a)
    c = state.acc_nv[vd] if accumulate else [[0] * len(w) for _ in range(rows)]
    state.acc_nv[vd] = nvfp4.matmul(a, sa, w, sw, c)


class VMATPUSH_WEIGHT_NV(
    MXUWeightPush,
    VRType[WeightBuffer, MatrixReg],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0001110,
):
    """``vmatpush.weight.nv w(vd), m(vs1)``: weight slot = m[vs1] and its scales (row j = output column j)."""
    def exec(self, state: ArchState) -> None:
        state.wb_nv[self.vd & 1][:] = state.mrf[self.vs1].view(torch.uint8)
        state.wb_nv_scale[self.vd & 1][:] = state.mrf_scale[self.vs1]


class VMATPUSH_ACC_BF16_NV(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0001111,
):
    """``vmatpush.acc.bf16.nv acc(vd), m(vs1)``: acc = {m[vs1], m[vs1+1]} (BF16, exact in FP32)."""
    def exec(self, state: ArchState) -> None:
        halves = (_nv_mreg_words(state, self.vs1), _nv_mreg_words(state, self.vs1 + 1))

        def bf16(r: int, j: int) -> int:  # column j: register j // 16, word (j % 16) // 2, half j % 2
            word = halves[j // 16][r][(j % 16) // 2]
            return (word >> 16 * (j % 2)) & 0xFFFF

        state.acc_nv[self.vd & 1] = [[bf16(r, j) << 16 for j in range(32)] for r in range(len(halves[0]))]


class VMATPUSH_ACC_FP32_NV(
    MXUAccumulatorPush,
    VRType[Accumulator, MatrixReg],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0010000,
):
    """``vmatpush.acc.fp32.nv acc(vd), m(vs1)``: acc = m[vs1..vs1+3], register q holding columns 8q..8q+7."""
    def exec(self, state: ArchState) -> None:
        regs = [_nv_mreg_words(state, self.vs1 + q) for q in range(4)]
        state.acc_nv[self.vd & 1] = [[regs[j // 8][r][j % 8] for j in range(32)] for r in range(len(regs[0]))]


class VMATPOP_BF16_ACC_NV(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0010001,
):
    """``vmatpop.bf16.acc.nv m(vd), acc(vs2)``: {m[vd], m[vd+1]} = BF16(acc), round to nearest even."""
    def exec(self, state: ArchState) -> None:
        from npu_model.configs import nvfp4
        acc = state.acc_nv[self.vs2 & 1]
        for half in range(2):
            rows = [[nvfp4.f32_to_bf16(acc[r][16 * half + 2 * k]) |
                     nvfp4.f32_to_bf16(acc[r][16 * half + 2 * k + 1]) << 16 for k in range(8)]
                    for r in range(len(acc))]
            _nv_write_mreg_words(state, self.vd + half, rows)


class VMATPOP_FP32_ACC_NV(
    MXUAccumulatorPop,
    VRType[MatrixReg, Accumulator],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0010010,
):
    """``vmatpop.fp32.acc.nv m(vd), acc(vs2)``: m[vd..vd+3] = acc, register q holding columns 8q..8q+7."""
    def exec(self, state: ArchState) -> None:
        acc = state.acc_nv[self.vs2 & 1]
        for q in range(4):
            _nv_write_mreg_words(state, self.vd + q, [row[8 * q:8 * q + 8] for row in acc])


class VMATMUL_NV(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0010011,
):
    """``vmatmul.nv acc(vd), m(vs1), w(vs2)``: acc = A @ W^T over 64 NVFP4 elements (FP32)."""
    def exec(self, state: ArchState) -> None:
        _nv_matmul(state, self.vd & 1, self.vs1, self.vs2 & 1, accumulate=False)


class VMATMUL_ACC_NV(
    MXUMatMul,
    VRType[Accumulator, WeightBuffer],
    exu=EXU.MATRIX_NVFP4,
    opcode=0b1110111,
    funct7=0b0010100,
):
    """``vmatmul.acc.nv acc(vd), m(vs1), w(vs2)``: acc += A @ W^T (one NVIDIA m16n8k64 step per element)."""
    def exec(self, state: ArchState) -> None:
        _nv_matmul(state, self.vd & 1, self.vs1, self.vs2 & 1, accumulate=True)


class VLOAD_SCALE(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b10):
    """``vload.scale m(vd), imm(x(rs1))``: m[vd]'s scale sidecar = 128 B of VMEM (128 B aligned)."""
    def exec(self, state: ArchState) -> None:
        address = _vmem_line_address(state, self.rs1, self.imm)
        state.mrf_scale[self.vd][:] = state.read_vmem(address, 0, state.mrf_scale[self.vd].numel())


class VSTORE_SCALE(TensorBaseOffset, VLSType, exu=EXU.LSU, opcode=0b0000111, funct2=0b11):
    """``vstore.scale m(vd), imm(x(rs1))``: 128 B of VMEM = m[vd]'s scale sidecar."""
    def exec(self, state: ArchState) -> None:
        address = _vmem_line_address(state, self.rs1, self.imm)
        state.write_vmem(address, 0, state.mrf_scale[self.vd].clone())


class DMA_LOAD(DMATransfer, RType, exu=EXU.DMA, opcode=0b1111011, funct3=None, funct7=0b0000000):
    """``dma.load x(rd), x(rs1), x(rs2), ch<N>``: DRAM ``{base, x[rs1]}`` -> VMEM word ``x[rd]``, ``x[rs2]`` bytes."""

    def exec(self, state: ArchState) -> None:
        length = state.read_xrf(self.rs2)
        data = state.read_dram(state.dma_dram_address(state.read_xrf(self.rs1)), length)
        state.write_vmem(state.dma_vmem_byte_address(state.read_xrf(self.rd)), 0, data)


class DMA_STORE(DMATransfer, RType, exu=EXU.DMA, opcode=0b1111011, funct3=None, funct7=0b0000001):
    """``dma.store x(rd), x(rs1), x(rs2), ch<N>``: VMEM word ``x[rs1]`` -> DRAM ``{base, x[rd]}``, ``x[rs2]`` bytes."""

    def exec(self, state: ArchState) -> None:
        length = state.read_xrf(self.rs2)
        data = state.read_vmem(state.dma_vmem_byte_address(state.read_xrf(self.rs1)), 0, length)
        state.write_dram(state.dma_dram_address(state.read_xrf(self.rd)), data)


class DMA_CONFIG(DMAConfig, RType, exu=EXU.DMA, opcode=0b1111111, funct3=None, funct7=0b0000000):
    """``dma.config x(rs1), ch<N>``: ``dma.base = x[rs1]``."""

    def exec(self, state: ArchState) -> None:
        state.base = state.read_xrf(self.rs1)


class DMA_WAIT(DMAWait, RType, exu=EXU.DMA, opcode=0b1111111, funct3=None, funct7=0b0000001):
    """``dma.wait ch<N>``: frontend fence until channel ``N`` is idle."""

    def exec(self, state: ArchState) -> None:
        pass
