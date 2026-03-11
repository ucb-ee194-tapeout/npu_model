from collections.abc import Iterable
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
    if isinstance(mnemonic, str):
        mnemonics = [mnemonic]
    elif isinstance(mnemonic, Iterable):
        mnemonics = list(mnemonic)
        if len(mnemonics) == 0 or not all(
            isinstance(alias, str) for alias in mnemonics
        ):
            raise TypeError(
                "@instr decorator iterable arguments must contain only strings"
            )
    else:
        raise TypeError("@instr decorator must be @instr(<your instruction>)")

    def effect(func: Callable) -> Callable:
        for alias in mnemonics:
            IsaSpec.operations[alias] = Operation(
                mnemonic=alias,
                instruction_type=instruction_type,
                effect=func,
            )

        return func

    return effect
