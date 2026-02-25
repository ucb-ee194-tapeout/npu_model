from typing import Callable


class InstructionType:
    SCALAR = 0
    VECTOR = 1
    MATRIX = 2
    DMA = 4
    BARRIER = 5
    MATRIX_SYSTOLIC = 6
    MATRIX_INNER = 7


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
