import re
from typing import Callable


_DMA_CHANNEL_RE = re.compile(r"\.ch(?P<channel>\d+)$")


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
        args: dict[str, int],
        delay: int = 0,
    ) -> None:
        self.mnemonic = mnemonic
        self.args = args
        self.delay = delay

    def dma_channel(self) -> int:
        match = _DMA_CHANNEL_RE.search(self.mnemonic)
        if match is not None:
            return int(match.group("channel"))
        if "flag" in self.args:
            return self.args["flag"]
        raise KeyError(f"DMA instruction '{self.mnemonic}' does not encode a channel")

    def __str__(self) -> str:
        display_mnemonic = self.mnemonic
        if display_mnemonic.startswith("dma.") and ".ch" not in display_mnemonic:
            try:
                display_mnemonic = f"{display_mnemonic}.ch{self.dma_channel()}"
            except KeyError:
                pass

        if self.mnemonic.startswith("dma.load.mxu") and "rd" in self.args:
            return (
                f"{display_mnemonic} w{self.args['rd']}, "
                f"{self.args['base']}, {self.args['size']}"
            )
        if self.mnemonic.startswith("dma.load") and "rd" in self.args:
            return (
                f"{display_mnemonic} m{self.args['rd']}, "
                f"{self.args['base']}, {self.args['size']}"
            )
        if self.mnemonic.startswith("dma.store") and "rs1" in self.args:
            return (
                f"{display_mnemonic} m{self.args['rs1']}, "
                f"{self.args['base']}, {self.args['size']}"
            )
        if self.mnemonic.startswith("dma.wait"):
            return display_mnemonic

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
