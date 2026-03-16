from typing import Callable, Optional
import inspect
import ast
from npu_model.software.instruction import Args
from dataclasses import dataclass
from npu_model.control.passes import register, mem, functional_unit


class Scalar:
    pass


class _R(Scalar):
    pass


class _I(Scalar):
    pass


class _B(Scalar):
    pass


class _J(Scalar):
    pass


class _U(Scalar):
    pass


class Matrix:
    pass


class _MatrixSystolic(Matrix):
    pass


class _MatrixIPT(Matrix):
    pass


class _Vector:
    pass


class _DMA:
    pass


class Transpose:
    pass


class _L(Transpose):
    pass


class _H(Transpose):
    pass


class _Barrier:
    pass


class _Delay:
    pass


# --- Instruction type namespace ---


class InstructionType:
    class SCALAR:
        R = _R()
        I = _I()
        B = _B()
        J = _J()
        U = _U()

    class MATRIX:
        MATRIX_SYSTOLIC = _MatrixSystolic()
        MATRIX_IPT = _MatrixIPT()

    VECTOR = _Vector()
    DMA = _DMA()
    DELAY = _Delay()
    BARRIER = _Barrier()

    class TRANSPOSE:
        H = _H()
        L = _L()


@dataclass
class ScalarArgs(Args):
    rd: int = 0
    rs1: int = 0
    rs2: int = 0
    imm: int = 0


@dataclass
class VectorArgs(Args):
    vrd: int = 0
    vrs1: int = 0
    vrs2: int = 0


@dataclass
class MatrixArgs(Args):
    mrd: int = 0
    mrs1: int = 0
    mrs2: int = 0


@dataclass
class DmaArgs(Args):
    rd: int = 0
    rs1: int = 0
    base: int = 0
    size: int = 0
    flag: int = 0


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
