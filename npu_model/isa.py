from __future__ import annotations
from typing import Callable, TypeAlias, TypeVar, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from npu_model.hardware.arch_state import ArchState

@dataclass
class ScalarArgs:
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0


@dataclass
class VectorArgs:
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    rs1: int = 0
    rs2: int = 0
    base: int = 0
    offset: int = 0
    imm12: int = 0
    imm: int = 0


@dataclass
class MatrixArgs:
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    rd: int = 0
    rs1: int = 0
    rs2: int = 0


@dataclass
class DmaArgs:
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    size: int = 0
    channel: int = 0  # FIXME: this is not a real arg. should also be renamed to channel (unless we just want to make 8 different insns)

# This bares some explanation if you're not familiar with Python's types
# Use "Args" when you mean to represent an input that can handle _all_ of the types.
# Use "ArgsT" when you mean to represent an input that can handle _any one_ of the types. 
Args: TypeAlias = ScalarArgs | VectorArgs | MatrixArgs | DmaArgs
ArgsT = TypeVar("ArgsT", ScalarArgs, VectorArgs, MatrixArgs, DmaArgs)

def _mask(val: int, bits: int):
    """returns the first `bits` bits of `val`"""
    return val & ((1 << bits) - 1)

class AsmInstructionType(ABC):
    mnemonics: list[str] = []

    @abstractmethod
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args) -> int:
        raise NotImplementedError()

    def add_mnemonic(self, mnemonic: str):
        self.mnemonics.append(mnemonic)

class _R(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, ScalarArgs) or isinstance(args, DmaArgs)):
            raise ValueError("Incorrect argument type specified.")

        funct7_b = _mask(funct7,   7)
        rd_b     = _mask(args.rd,  5)
        rs1_b    = _mask(args.rs1, 5)
        rs2_b    = _mask(args.rs2, 5)
        opcode_b = _mask(opcode,   7)
        if isinstance(args,DmaArgs):
            # FIXME: related to the above fixme in DmaArgs — set funct3 to channel for now
            funct3_b = _mask(args.channel, 3)
        else:
            funct3_b = _mask(funct3, 3)

        return (funct7_b << 25) | (rs2_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (rd_b << 7) | opcode_b
    pass

class _I(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, ScalarArgs) or isinstance(args, DmaArgs)):
            raise ValueError("Incorrect argument type specified.")

        rd_b     = _mask(args.rd,  5)
        rs1_b    = _mask(args.rs1, 5)
        opcode_b = _mask(opcode,  7)
        if isinstance(args,DmaArgs):
            # FIXME: related to the above fixme in DmaArgs — set funct3 to channel for now
            funct3_b = _mask(args.channel, 3)
            imm_b    = _mask(args.size, 12)
        else:
            funct3_b = _mask(funct3,   3)
            imm_b    = _mask(args.imm, 12)

        return (imm_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (rd_b << 7) | opcode_b
    pass

class _S(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, ScalarArgs)):
            raise ValueError("Incorrect argument type specified.")
        
        imm_b    = _mask(args.imm, 12)
        imm1_b   = imm_b & 0b000011111111
        imm2_b   = (imm_b & 0b111100000000) >> 8

        rs2_b    = _mask(args.rs2, 5)
        rs1_b    = _mask(args.rs1, 5)
        funct3_b = _mask(funct3,   3)
        opcode_b = _mask(opcode,   7)

        return (imm1_b << 24) | (rs2_b << 19) | (rs1_b << 14) | (funct3_b << 11) | (imm2_b << 7) | opcode_b
    pass

class _SB(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, ScalarArgs)):
            raise ValueError("Incorrect argument type specified.")
        
        imm_b    = _mask(args.imm, 13)
        imm12_b  = (imm_b >> 12) & 1
        imm49_b  = (imm_b >> 5)  & 0x3F
        imm04_b  = (imm_b >> 1)  & 0xF
        imm11_b  = (imm_b >> 11) & 1
        rs2_b    = _mask(args.rs2, 5)
        rs1_b    = _mask(args.rs1, 5)
        funct3_b = _mask(funct3,   3)
        opcode_b = _mask(opcode,   7)

        return (imm12_b << 31) | (imm49_b << 25) | (rs2_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (imm04_b << 8) | (imm11_b << 4) | opcode_b
    pass

class _U(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, ScalarArgs)):
            raise ValueError("Incorrect argument type specified.")

        imm_b    = _mask(args.imm, 20)
        rd_b     = _mask(args.rd,  5)
        opcode_b = _mask(opcode,   7)

        return (imm_b << 12) | (rd_b << 7) | opcode_b
    pass

class _UJ(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not isinstance(args, ScalarArgs):
            raise ValueError("Incorrect argument type specified.")
        
        imm_b     = _mask(args.imm, 21)
        imm19_b   = (imm_b >> 20) & 1
        imm110_b  = (imm_b >> 1) & 0x3ff
        imm11_b   = (imm_b >> 11) & 1
        imm1219_b = (imm_b >> 12) & 0xff
        rd_b      = _mask(args.rd,  5)
        opcode_b  = _mask(opcode,   7)
        
        return (imm19_b << 31) | (imm110_b << 21) | (imm11_b << 20) | (imm1219_b << 12) | (rd_b << 7) | opcode_b
    pass

class _VLS(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not isinstance(args, VectorArgs):
            raise ValueError("Incorrect argument type specified.")
        
        imm_b    = _mask(args.imm,  12)
        rs1_b    = _mask(args.rs1,  5)
        funct2_b = _mask(funct2,    2)
        vd_b     = _mask(args.vd,   6)
        opcode_b = _mask(opcode,    7)

        return (imm_b << 20) | (rs1_b << 15) | (funct2_b << 13) | (vd_b << 7) | opcode_b
    pass

class _VR(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not (isinstance(args, VectorArgs) or isinstance(args, MatrixArgs)):
            raise ValueError("Incorrect argument type specified.")
        
        funct7_b = _mask(funct7,   7)
        vs2_b    = _mask(args.vs2, 6)
        vs1_b    = _mask(args.vs1, 6)
        vd_b     = _mask(args.vd,  6)
        opcode_b = _mask(opcode,   7)

        return (funct7_b << 25) | (vs2_b << 19) | (vs1_b << 13) << (vd_b << 7) | opcode_b
    pass

class _VI(AsmInstructionType):
    def assemble(self, opcode: int, funct2: int, funct3: int, funct7: int, args: Args):
        if not isinstance(args, VectorArgs):
            raise ValueError("Incorrect argument type specified.")
        
        imm_b    = _mask(args.imm,  16)
        funct3_b = _mask(funct3,    2)
        vd_b     = _mask(args.vd,   6)
        opcode_b = _mask(opcode,    7)
        
        return (imm_b << 16) | (funct3_b << 13) | (vd_b << 7) | opcode_b
    pass

# --- Instruction type namespace ---

class InstructionType:
    class SCALAR:
        R = _R()
        I = _I()
        S = _S()
        SB = _SB()
        U = _U()
        UJ = _UJ()

    class VECTOR:
        VLS = _VLS()
        VR = _VR()
        VI = _VI()

    class DMA:
        R = _R()
        I = _I()

    class MATRIX_SYSTOLIC:
        VR = _VR()

    class MATRIX_IPT:
        VR = _VR()

    class DELAY:
        I = _I()

    class BARRIER:
        I = _I()


class Operation:
    def __init__(
        self,
        mnemonic: str,
        instruction_type: AsmInstructionType,
        opcode: int,
        funct2: int,
        funct3: int,
        funct7: int,
        effect: Callable[[ArchState,ArgsT],None],
    ) -> None:
        self.mnemonic = mnemonic
        self.instruction_type = instruction_type
        self.effect = effect
        self.opcode = opcode
        self.funct2 = funct2
        self.funct3 = funct3
        self.funct7 = funct7

    def __str__(self) -> str:
        return self.mnemonic


class IsaSpec:
    operations: dict[str, Operation] = {}


# Global registry accumulating decode table rows
# FIXME: Not typed but idk the type.
_decode_table: list[dict] = []

# Column order — matches the signal list in AtlasCtrlSigs.decode()
_COL_HEADERS = [
    "valid",
    "br_type",
    "src1",
    "src2",
    "dst",
    "msrc1",
    "msrc2",
    "mdst",
    "mem_read",
    "mem_write",
    "wb",
    "pc",
    "mxu_0_valid",
    "mxu_1_valid",
    "scalar_valid",
    "vpu_valid",
    "xlu_valid",
    "dma_valid",
    "alu_op",
    "vpu_op",
    "xlu_op",
]

# Default row — emitted by AtlasCtrlSigs.default
# Analogous to IntCtrlSigs.default in Rocket
_DEFAULT = {
    "valid": "N",
    "br_type": "BR_X",
    "src1": "X",
    "src2": "X",
    "dst": "X",
    "msrc1": "X",
    "msrc2": "X",
    "mdst": "X",
    "mem_read": "N",
    "mem_write": "N",
    "wb": "X",
    "pc": "X",
    "mxu_0_valid": "N",
    "mxu_1_valid": "N",
    "scalar_valid": "N",
    "vpu_valid": "N",
    "xlu_valid": "N",
    "dma_valid": "N",
    "alu_op": "ALU_OP_X",
    "vpu_op": "VPU_OP_X",
    "xlu_op": "XLU_OP_X",
}


def instr(mnemonic: str | None, instruction_type: AsmInstructionType, opcode: int, funct2: int = 0, funct3: int = 0, funct7: int = 0):
    if not isinstance(mnemonic, str):
        raise TypeError("@instr decorator must be @instr(<your instruction>)")

    def effect(func: Callable[[ArchState,ArgsT],None]) -> Callable[[ArchState,ArgsT],None]:
        instruction_type.add_mnemonic(mnemonic)
        IsaSpec.operations[mnemonic] = Operation(
            mnemonic=mnemonic,
            instruction_type=instruction_type,
            effect=func,
            opcode=opcode,
            funct2=funct2,
            funct3=funct3,
            funct7=funct7
        )

        return func

    return effect
