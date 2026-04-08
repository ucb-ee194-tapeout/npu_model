from typing import Callable, Generic, TypeGuard, Any
from dataclasses import asdict, is_dataclass

from ..isa import IsaSpec, ScalarArgs, DmaArgs, VectorArgs, MatrixArgs, ArgsT
from npu_model.hardware.arch_state import ArchState

class Instruction(Generic[ArgsT]):
    """
    An instruction in the program sequence.

    Attributes:
        id: Unique instruction ID
        mnemonic: The mnemonic of the instruction
        args: The arguments of the instruction
        delay: The delay of the instruction
    """

    def __init__(
        self,
        mnemonic: str,
        args: ArgsT,
        delay: int = 0,
    ) -> None:
        self.mnemonic = mnemonic
        self.args = args
        self.delay = delay

    def __str__(self) -> str:
        args_dict = asdict(self.args) if is_dataclass(self.args) else self.args
        args_str = [f"{k}={v}" for k, v in args_dict.items()]
        return f"{self.mnemonic} {', '.join(args_str)}"

    def assemble(self) -> int:
        # find type from mnemonic table
        operation = IsaSpec.operations[self.mnemonic]

        # our first pass should correctly set these things in the IR,
        # but as a convenience feature for people writing in the IR
        # we fix args for shifts and breakpoint/ecall here
        if self.mnemonic == "ecall" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000000
        elif self.mnemonic == "ebreak" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000001
        elif (self.mnemonic == "srli" or self.mnemonic == "srai") and isinstance(self.args, ScalarArgs):
            self.args.imm = self.args.imm & 0b000000011111
        elif self.mnemonic == "srai" and isinstance(self.args, ScalarArgs):
            self.args.imm = (self.args.imm & 0b000000011111) | 0b0100000000000
        elif self.mnemonic == "fence" and isinstance(self.args, ScalarArgs):
            self.args.imm = 0b000000000000
        
        if isinstance(self.args, VectorArgs)and self.args.imm12 != 0:
            self.args.imm = self.args.imm12

        return operation.instruction_type.assemble(
            operation.opcode,
            operation.funct2,
            operation.funct3,
            operation.funct7,
            self.args
        )

class Uop(Generic[ArgsT]):
    """
    A dynamic instruction instance that is executing in the simulation
    """

    _next_id: int = 0

    def __init__(self, insn: Instruction[ArgsT]) -> None:
        self.id = Uop._next_id
        Uop._next_id += 1
        self.insn = insn

        self.dispatch_delay: int = 0
        """the number of dispatch stalling cycles left"""
        self.execute_delay: int = 0
        """the number of execute stalling cycles left"""

        self.execute_fn: Callable[[ArchState,ArgsT],None] | None = None

def is_scalar_uop(uop: Uop[Any]) -> TypeGuard[Uop[ScalarArgs]]:
    return isinstance(uop.insn.args, ScalarArgs)

def is_dma_uop(uop: Uop[Any]) -> TypeGuard[Uop[DmaArgs]]:
    return isinstance(uop.insn.args, DmaArgs)

def is_vector_uop(uop: Uop[Any]) -> TypeGuard[Uop[VectorArgs]]:
    return isinstance(uop.insn.args, VectorArgs)

def is_matrix_uop(uop: Uop[Any]) -> TypeGuard[Uop[MatrixArgs]]:
    return isinstance(uop.insn.args, MatrixArgs)