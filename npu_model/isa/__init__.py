from typing import TYPE_CHECKING, TypeIs, Any, ClassVar, cast
from abc import ABC, abstractmethod
from npu_model.isa_types import *
from npu_model.isa.table import load
if TYPE_CHECKING:
    from npu_model.hardware.arch_state import ArchState

TABLE = load()
"""The ISA table (atlas_isa.toml). Each instruction class below takes its encoding from it."""

ISA_VERSION = TABLE.version
"""Atlas ISA (major, minor): bits 31:24 and 23:16 of CSR 0xCC0 (Instructions.ISA_MAJOR / ISA_MINOR in
the RTL), the highest `ver` of the table. A change of instruction encoding or semantics increments it."""

EXU_OF_UNIT = {"scalar": EXU.SCALAR, "lsu": EXU.LSU, "vpu": EXU.VECTOR, "xlu": EXU.XLU,
               "mxu1": EXU.MATRIX_INNER, "nv": EXU.MATRIX_NVFP4, "dma": EXU.DMA}

# Utility functions for assembly
def _mask(val: int, bits: int):
    """returns the first `bits` bits of `val`"""
    return val & ((1 << bits) - 1)

class IsaSpec:
    operations: dict[str,type[Instruction]] = {}

class Instruction(ABC):
    mnemonic: str  = NotImplemented
    opcode: Opcode = NotImplemented
    exu: EXU       = NotImplemented
    fmt: ClassVar[str]
    """The instruction type's format in the ISA table."""

    def __str__(self):
        values = [str(v) for v in self.__dict__.values()]
        return f"{self.mnemonic} {', '.join(values)}"

    @abstractmethod
    def to_bytecode(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def exec(
        self, state: ArchState
    ) -> None:
        raise NotImplementedError()

    def jump_target(self, state: ArchState) -> int | None:
        """The byte address that the instruction jumps to, or None if it does not jump."""
        return None

    def __init_subclass__(cls, instr: bool = True) -> None:
        """An instruction class takes its mnemonic from its name (VREDSUM_ROW_BF16 gives
        vredsum.row.bf16), and its opcode, selector and fixed fields and unit from the mnemonic's table
        entry."""
        super().__init_subclass__()
        if not instr:
            return
        cls.mnemonic = cls.__name__.lower().replace("_", ".")
        entry = TABLE.entries[cls.mnemonic]
        assert entry.format == cls.fmt, f"{cls.mnemonic}: the table has format {entry.format}, the class {cls.fmt}"
        cls.opcode = Opcode(entry.opcode)
        cls.exu = EXU_OF_UNIT[entry.unit]
        for field, value in (entry.select | entry.fixed).items():
            setattr(cls, field, value)
        IsaSpec.operations[cls.mnemonic] = cls

class Word(Instruction, instr=False):
    """A word that is no instruction (`.word` in assembly), for example a v1 word or a word with a
    reserved bit set. The core halts at it: it is illegal."""
    mnemonic = ".word"
    exu = EXU.SCALAR

    def __init__(self, word: int) -> None:
        if not 0 <= word < 1 << 32:
            raise ValueError(f".word {word:#x} is not a 32-bit word")
        if (entry := TABLE.decode(word)) is not None:
            raise ValueError(f".word {word:#010x} is a {entry.mnemonic} instruction: write the instruction")
        self.word = word

    def to_bytecode(self) -> int:
        return self.word

    def exec(self, state: ArchState) -> None:
        raise RuntimeError(f"Illegal instruction word {self.word:#010x}")

class RType(Instruction, instr=False):
    fmt = "R"
    funct3: Funct3 = NotImplemented
    funct7: Funct7 = NotImplemented
    rd: ScalarReg  = ScalarReg(0)
    rs1: ScalarReg = ScalarReg(0)
    rs2: ScalarReg = ScalarReg(0)


    def to_bytecode(self):
        funct7_b = _mask(self.funct7, 7)
        rd_b = _mask(self.rd, 5)
        rs1_b = _mask(self.rs1, 5)
        rs2_b = _mask(self.rs2, 5)
        opcode_b = _mask(self.opcode, 7)
        funct3_b = _mask(self.funct3, 3)

        return (
            (funct7_b << 25)
            | (rs2_b << 20)
            | (rs1_b << 15)
            | (funct3_b << 12)
            | (rd_b << 7)
            | opcode_b
        )

class IType[RD: (ScalarReg, ExponentReg) = ScalarReg, IMM: (Imm12, SBImm12) = Imm12](Instruction, instr=False):
    fmt = "I"
    funct3: Funct3         = NotImplemented
    rd: RD
    rs1: ScalarReg         = ScalarReg(0)
    imm: IMM

    def to_bytecode(self):
        rd = self.rd if hasattr(self, 'rd') else 0
        imm = self.imm if hasattr(self, 'imm') else 0
        rd_b = _mask(rd, 5)
        rs1_b = _mask(self.rs1, 5)
        opcode_b = _mask(self.opcode, 7)
        imm_b = _mask(imm, 12)
        funct3_b = _mask(self.funct3, 3)

        return (imm_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (rd_b << 7) | opcode_b

class SType(Instruction, instr=False):
    fmt = "S"
    funct3: Funct3    = NotImplemented
    rs1: ScalarReg    = ScalarReg(0)
    rs2: ScalarReg    = ScalarReg(0)
    imm: Imm12        = Imm12(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 12)

        # RISC-V Standard Split:
        # imm_high = bits 11 down to 5
        # imm_low  = bits 4 down to 0
        imm_high = (imm_b >> 5) & 0x7F
        imm_low = imm_b & 0x1F

        rs2_b = _mask(self.rs2, 5)
        rs1_b = _mask(self.rs1, 5)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)

        return (
            (imm_high << 25)
            | (rs2_b << 20)
            | (rs1_b << 15)
            | (funct3_b << 12)
            | (imm_low << 7)
            | opcode_b
        )

class SBType(Instruction, instr=False):
    fmt = "B"
    funct3: Funct3    = NotImplemented
    rs1: ScalarReg    = ScalarReg(0)
    rs2: ScalarReg    = ScalarReg(0)
    imm: SBImm12      = SBImm12(0)

    def to_bytecode(self):
        # Branch immediates are 13 bits (bit 0 is always 0)
        imm_b = _mask(self.imm, 13)
        
        # Extract bits exactly per the RISC-V ISA spec
        imm12    = (imm_b >> 12) & 0x1    # bit 12
        imm11    = (imm_b >> 11) & 0x1    # bit 11
        imm10_5  = (imm_b >> 5)  & 0x3F   # bits 10:5 (6 bits)
        imm4_1   = (imm_b >> 1)  & 0xF    # bits 4:1  (4 bits)

        rs2_b    = _mask(self.rs2, 5)
        rs1_b    = _mask(self.rs1, 5)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)

        return (
            (imm12 << 31)    |
            (imm10_5 << 25)  |
            (rs2_b << 20)    |
            (rs1_b << 15)    |
            (funct3_b << 12) |
            (imm4_1 << 8)    | # bits 4:1 go to bits 11:8
            (imm11 << 7)     | # bit 11 goes to bit 7
            opcode_b
        )

class UType(Instruction, instr=False):
    fmt = "U"
    rd: ScalarReg = ScalarReg(0)
    imm: Imm20    = Imm20(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 20)
        rd_b = _mask(self.rd, 5)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 12) | (rd_b << 7) | opcode_b

class UJType(Instruction, instr=False):
    fmt = "J"
    rd: ScalarReg = ScalarReg(0)
    imm: Imm21    = Imm21(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 21)
        imm19_b = (imm_b >> 20) & 1
        imm110_b = (imm_b >> 1) & 0x3FF
        imm11_b = (imm_b >> 11) & 1
        imm1219_b = (imm_b >> 12) & 0xFF
        rd_b = _mask(self.rd, 5)
        opcode_b = _mask(self.opcode, 7)

        return (
            (imm19_b << 31)
            | (imm110_b << 21)
            | (imm11_b << 20)
            | (imm1219_b << 12)
            | (rd_b << 7)
            | opcode_b
        )

class VLSType(Instruction, instr=False):
    fmt = "VLS"
    funct2: Funct2 = NotImplemented
    vd: MatrixReg  = MatrixReg(0)
    rs1: ScalarReg = ScalarReg(0)
    imm: Imm12     = Imm12(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 12)
        rs1_b = _mask(self.rs1, 5)
        funct2_b = _mask(self.funct2, 2)
        vd_b = _mask(self.vd, 6)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 20) | (rs1_b << 15) | (funct2_b << 13) | (vd_b << 7) | opcode_b

class VRType[VD: (MatrixReg,Accumulator,WeightBuffer) = MatrixReg, VS2: (MatrixReg,Accumulator,WeightBuffer) = MatrixReg](Instruction, instr=False):
    fmt = "VR"
    funct7: Funct7    = NotImplemented
    vs1: MatrixReg
    es1: ExponentReg
    vs2: VS2
    rs2: ScalarReg
    vd:  VD

    def to_bytecode(self):
        vs1 = self.vs1 if hasattr(self, 'vs1') else (self.es1 if hasattr(self, 'es1') else 0)
        # A .vx instruction holds rs2 in bits 24:20, the upper 5 bits of vs2.
        vs2 = self.vs2 if hasattr(self, 'vs2') else (self.rs2 << 1 if hasattr(self, 'rs2') else 0)
        vd  = self.vd  if hasattr(self, 'vd')  else 0

        funct7_b = _mask(self.funct7, 7) # Bits 31:25
        vs2_b    = _mask(vs2, 6)          # Bits 24:19 (6 bits, ScalarDecoder vr_vs2)
        vs1_b    = _mask(vs1, 6)          # Bits 18:13 (6 bits, vr_vs1)
        vd_b     = _mask(vd, 6)           # Bits 12:7  (6 bits, vr_vd)
        opcode_b = _mask(self.opcode, 7)  # Bits 6:0

        return (
            (funct7_b << 25) |
            (vs2_b << 19)    |
            (vs1_b << 13)    |
            (vd_b << 7)      | 
            opcode_b
        )
        

class VIType(Instruction, instr=False):
    fmt = "VI"
    funct3: Funct3 = NotImplemented
    vd: MatrixReg  = MatrixReg(0)
    imm: Imm16     = Imm16(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 16)
        funct3_b = _mask(self.funct3, 3)
        vd_b = _mask(self.vd, 6)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 16) | (funct3_b << 13) | (vd_b << 7) | opcode_b

class CSRType(Instruction, instr=False):
    fmt = "I"
    funct3: Funct3 = NotImplemented
    rs1: ScalarReg = ScalarReg(0)
    rd: ScalarReg  = ScalarReg(0)
    imm: Imm12     = Imm12(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 12)
        rs1_b = _mask(self.rs1, 5)
        funct3_b = _mask(self.funct3, 3)
        rd_b = _mask(self.rd, 5)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (rd_b << 7) | opcode_b

    @property
    def writes(self) -> bool:
        """csrrw and csrrwi write the CSR; the set and clear forms only when rs1 (or uimm) is not 0."""
        return self.funct3 & 0b11 == 0b01 or self.rs1 != 0


def is_scalar_reg(obj: Any) -> TypeIs[ScalarReg]:
    return isinstance(obj, ScalarReg)

def is_exponent_reg(obj: Any) -> TypeIs[ScalarReg]:
    return isinstance(obj, ExponentReg)

def is_scalar_itype(insn: Instruction) -> TypeIs[IType[ScalarReg]]:
    return isinstance(insn, IType) and (not hasattr(cast(IType[Any], insn), 'rd') or is_scalar_reg(cast(IType[Any], insn).rd))

def is_exponent_itype(insn: Instruction) -> TypeIs[IType[ExponentReg]]:
    return isinstance(insn, IType) and hasattr(cast(IType[Any], insn), 'rd') and is_exponent_reg(cast(IType[Any], insn).rd)
