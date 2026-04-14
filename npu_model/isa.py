from typing import TYPE_CHECKING, TypeIs, Any, cast
from abc import ABC, abstractmethod
from npu_model.isa_types import *
if TYPE_CHECKING:
    from npu_model.hardware.arch_state import ArchState

# Utility functions for assembly
def _mask(val: int, bits: int):
    """returns the first `bits` bits of `val`"""
    return val & ((1 << bits) - 1)

class IsaSpec:
    operations: dict[str,type[Instruction]] = {}
    R: dict[str,type[RType]] = {}
    I: dict[str,type[IType[Any]]] = {}
    S: dict[str,type[SType]] = {}
    SB: dict[str,type[SBType]] = {}
    U: dict[str,type[UType]] = {}
    UJ: dict[str,type[UJType]] = {}
    VLS: dict[str,type[VLSType]] = {}
    VR: dict[str,type[VRType[Any,Any]]] = {}
    VI: dict[str,type[VIType]] = {}
    CSR: dict[str,type[CSRType]] = {}

class Instruction(ABC):
    mnemonic: str  = NotImplemented
    opcode: Opcode = NotImplemented
    exu: EXU       = NotImplemented

    def __str__(self):
        values = [str(v) for v in self.__dict__.values()]
        return f"{self.mnemonic}: {', '.join(values)}"

    @abstractmethod
    def to_bytecode(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def exec(
        self, state: ArchState
    ) -> None:
        raise NotImplementedError()

    def __init_subclass__(cls, exu: EXU | None = None, mnemonic: str | None = None, opcode: OpcodeL | None = None, instr: bool = True) -> None:
        if exu:
            cls.exu = exu
        
        if opcode:
            cls.opcode = Opcode(opcode)

        if mnemonic:
            cls.mnemonic = mnemonic
        
        if instr:
            IsaSpec.operations[cls.mnemonic] = cls
        return super().__init_subclass__()

class RType(Instruction, instr=False):
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

    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, funct7: Funct7L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        cls.funct7 = Funct7(funct7)
        IsaSpec.R[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class IType[Reg: (ScalarReg, ExponentReg) = ScalarReg](Instruction, instr=False):
    funct3: Funct3         = NotImplemented
    rd: Reg
    rs1: ScalarReg         = ScalarReg(0)
    size: Imm12            = Imm12(0)
    imm: Imm12             = Imm12(0)

    def to_bytecode(self):
        rd = self.rd if hasattr(self, 'rd') else 0
        rd_b = _mask(rd, 5)
        rs1_b = _mask(self.rs1, 5)
        opcode_b = _mask(self.opcode, 7)
        imm_b = _mask(self.imm, 12)
        funct3_b = _mask(self.funct3, 3)

        return (imm_b << 20) | (rs1_b << 15) | (funct3_b << 12) | (rd_b << 7) | opcode_b
    
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        IsaSpec.I[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)
    
class SType(Instruction, instr=False):
    funct3: Funct3    = NotImplemented
    rs1: ScalarReg    = ScalarReg(0)
    rs2: ScalarReg    = ScalarReg(0)
    imm: Imm12        = Imm12(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 12)
        imm1_b = imm_b & 0b000011111111
        imm2_b = (imm_b & 0b111100000000) >> 8

        rs2_b = _mask(self.rs2, 5)
        rs1_b = _mask(self.rs1, 5)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)

        return (
            (imm1_b << 24)
            | (rs2_b << 19)
            | (rs1_b << 14)
            | (funct3_b << 11)
            | (imm2_b << 7)
            | opcode_b
        )
    
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        IsaSpec.S[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class SBType(Instruction, instr=False):
    funct3: Funct3    = NotImplemented
    rs1: ScalarReg    = ScalarReg(0)
    rs2: ScalarReg    = ScalarReg(0)
    imm: SBImm12      = SBImm12(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 13)
        imm12_b = (imm_b >> 12) & 1
        imm49_b = (imm_b >> 5) & 0x3F
        imm04_b = (imm_b >> 1) & 0xF
        imm11_b = (imm_b >> 11) & 1
        rs2_b = _mask(self.rs2, 5)
        rs1_b = _mask(self.rs1, 5)
        funct3_b = _mask(self.funct3, 3)
        opcode_b = _mask(self.opcode, 7)

        return (
            (imm12_b << 31)
            | (imm49_b << 25)
            | (rs2_b << 20)
            | (rs1_b << 15)
            | (funct3_b << 12)
            | (imm04_b << 8)
            | (imm11_b << 4)
            | opcode_b
        )
    
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        IsaSpec.SB[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class UType(Instruction, instr=False):
    rd: ScalarReg = ScalarReg(0)
    imm: Imm20    = Imm20(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 20)
        rd_b = _mask(self.rd, 5)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 12) | (rd_b << 7) | opcode_b

    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        IsaSpec.U[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class UJType(Instruction, instr=False):
    rd: ScalarReg = ScalarReg(0)
    imm: Imm20    = Imm20(0)

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
    
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        IsaSpec.UJ[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)
    
class VLSType(Instruction, instr=False):
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
    
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct2: Funct2L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct2 = Funct2(funct2)
        IsaSpec.VLS[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class VRType[VD: (MatrixReg,Accumulator,WeightBuffer) = MatrixReg, VS2: (MatrixReg,Accumulator,WeightBuffer) = MatrixReg](Instruction, instr=False):
    funct7: Funct7    = NotImplemented
    vs1: MatrixReg
    es1: ExponentReg
    vs2: VS2
    vd:  VD

    def to_bytecode(self):
        vs1 = self.vs1 if hasattr(self, 'vs1') else (self.es1 if hasattr(self, 'es1') else 0)
        vs2 = self.vs2 if hasattr(self, 'vs2') else 0
        vd  = self.vd if hasattr(self, 'vd') else 0

        funct7_b = _mask(self.funct7, 7)
        vs2_b = _mask(vs2, 6)
        vs1_b = _mask(vs1, 6)
        vd_b = _mask(vd, 6)
        opcode_b = _mask(self.opcode, 7)

        return (
            (funct7_b << 25) | (vs2_b << 19) | (vs1_b << 13) << (vd_b << 7) | opcode_b
        )
        
    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct7: Funct7L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct7 = Funct7(funct7)
        IsaSpec.VR[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class VIType(Instruction, instr=False):
    funct3: Funct3 = NotImplemented
    vd: MatrixReg  = MatrixReg(0)
    imm: Imm16     = Imm16(0)

    def to_bytecode(self):
        imm_b = _mask(self.imm, 16)
        funct3_b = _mask(self.funct3, 2)
        vd_b = _mask(self.vd, 6)
        opcode_b = _mask(self.opcode, 7)

        return (imm_b << 16) | (funct3_b << 13) | (vd_b << 7) | opcode_b

    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        IsaSpec.VI[mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)

class CSRType(Instruction, instr=False):
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

    def __init_subclass__(cls, exu: EXU, opcode: OpcodeL, funct3: Funct3L, mnemonic: str | None = None) -> None:
        mnemonic = mnemonic if mnemonic != None else cls.__name__.lower().replace("_",".")
        cls.funct3 = Funct3(funct3)
        IsaSpec.CSR[cls.mnemonic] = cls
        return super().__init_subclass__(exu, mnemonic, opcode, instr=True)


def is_scalar_reg(obj: Any) -> TypeIs[ScalarReg]:
    return isinstance(obj, ScalarReg)

def is_exponent_reg(obj: Any) -> TypeIs[ScalarReg]:
    return isinstance(obj, ExponentReg)

def is_scalar_itype(insn: Instruction) -> TypeIs[IType[ScalarReg]]:
    return isinstance(insn, IType) and (not hasattr(cast(IType[Any], insn), 'rd') or is_scalar_reg(cast(IType[Any], insn).rd))

def is_exponent_itype(insn: Instruction) -> TypeIs[IType[ExponentReg]]:
    return isinstance(insn, IType) and hasattr(cast(IType[Any], insn), 'rd') and is_exponent_reg(cast(IType[Any], insn).rd)

# Global registry accumulating decode table rows
# FIXME: Not typed but idk the type.
_decode_table: list[dict[Any, Any]] = []

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