from typing import Dict, Callable


class Instruction:
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
        args: Dict[str, int],
        delay: int = 0,
    ) -> None:
        self.mnemonic = mnemonic
        self.args = args
        self.delay = delay

    def __str__(self) -> str:
        args_str = []
        for key, value in self.args.items():
            args_str.append(f"{key}={value}")
        return f"{self.mnemonic} {', '.join(args_str)}"


class Uop:
    """
    A dynamic instruction instance that is executing in the simulation
    """

    _next_id: int = 0

    def __init__(self, insn: Instruction) -> None:
        self.id = Uop._next_id
        Uop._next_id += 1
        self.insn = insn

        self.dispatch_delay: int = 0
        """the number of dispatch stalling cycles left"""
        self.execute_delay: int = 0
        """the number of execute stalling cycles left"""

        self.execute_fn: Callable | None = None
