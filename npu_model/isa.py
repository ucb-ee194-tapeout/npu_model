from typing import Callable
from npu_model.software.instruction import Args
from dataclasses import dataclass


class _Scalar:
    pass

class _MatrixSystolic:
    pass

class _MatrixIPT:
    pass

class _Vector:
    pass

class _DMA:
    pass

class _Delay:
    pass

class _Barrier:
    pass

# --- Instruction type namespace ---


class InstructionType:
    class R:
        SCALAR = _Scalar()
        DMA = _DMA()
    
    class I:
        SCALAR = _Scalar()
        DMA = _DMA()
        BARRIER = _Barrier()
        DELAY = _Delay()
    
    class S:
        SCALAR = _Scalar()
    
    class SB:
        SCALAR = _Scalar()
    
    class U:
        SCALAR = _Scalar()
    
    class UJ:
        SCALAR = _Scalar()
    
    class VLS:
        VECTOR = _Vector()

    class VR:
        VECTOR = _Vector()
        MATRIX_SYSTOLIC = _MatrixSystolic()
        MATRIX_IPT = _MatrixIPT()

    class VI:
        VECTOR = _Vector()


@dataclass
class ScalarArgs(Args):
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0


@dataclass
class VectorArgs(Args):
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
class MatrixArgs(Args):
    vd: int = 0
    vs1: int = 0
    vs2: int = 0
    rd: int = 0
    rs1: int = 0
    rs2: int = 0


@dataclass
class DmaArgs(Args):
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    size: int = 0
    flag: int = 0  # FIXME: this is not a real arg. should also be renamed to channel (unless we just want to make 8 different insns)


class Operation:
    def __init__(
        self,
        mnemonic: str,
        instruction_type: InstructionType,
        effect: Callable,
    ) -> None:
        self.mnemonic = mnemonic
        self.instruction_type = instruction_type
        self.effect = effect

    def __str__(self) -> str:
        return self.mnemonic


class IsaSpec:
    operations: dict[str, Operation] = {}


# Global registry accumulating decode table rows
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


def instr(mnemonic, instruction_type: InstructionType):
    if not isinstance(mnemonic, str):
        raise TypeError("@instr decorator must be @instr(<your instruction>)")

    def effect(func: Callable) -> Callable:
        IsaSpec.operations[mnemonic] = Operation(
            mnemonic=mnemonic,
            instruction_type=instruction_type,
            effect=func,
        )

        return func

    return effect
