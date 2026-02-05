from typing import Optional, Tuple
import numpy as np
from ..logging.logger import Logger


class ArchState:
    def __init__(
        self,
        logger: Optional[Logger] = None,
        matrix_shape: Tuple[int, int] = (8, 128),
        memory_size: int = 1024,
    ) -> None:
        # metadata
        self.matrix_shape = matrix_shape
        self.memory_size = memory_size
        # state
        self.mem: bytearray = bytearray()
        self.xrf: list[int] = []
        self.mrf: list[np.ndarray] = []
        self.flags: list[bool] = []
        self.pc: int = 0
        self.npc: int = 0
        self.logger = logger

    def reset(self) -> None:
        self.mem = bytearray(self.memory_size)
        self.xrf = [0] * 32
        self.mrf = [np.zeros(self.matrix_shape) for _ in range(32)]
        self.flags = [False] * 3
        self.pc = 0
        self.npc = 0

    def set_xrf(self, rd: int, value: int) -> None:
        if rd == 0:
            return
        if rd < len(self.xrf) and self.xrf[rd] == value:
            return
        self.xrf[rd] = value
        if self.logger is not None:
            self.logger.log_arch_value("xrf", rd, value)

    def set_npc(self, value: int) -> None:
        self.npc = value

    def set_pc(self, value: int) -> None:
        if self.pc == value:
            return
        self.pc = value
        if self.logger is not None:
            self.logger.log_arch_value("pc", 0, value)

    def write_bytes(self, base: int, data: bytes) -> None:
        print(f"Writing {len(data)} bytes to memory at base {base}")
        assert (
            base + len(data) <= self.memory_size
        ), f"Memory write out of bounds: {base} + {len(data)} > {self.memory_size}"
        self.mem[base : base + len(data)] = data

    def read_bytes(self, base: int, length: int) -> bytes:
        assert (
            base + length <= self.memory_size
        ), f"Memory read out of bounds: {base} + {length} > {self.memory_size}"
        return self.mem[base : base + length]

    def set_flag(self, flag: int) -> None:
        self.flags[flag] = True

    def clear_flag(self, flag: int) -> None:
        self.flags[flag] = False

    def check_flag(self, flag: int) -> bool:
        return self.flags[flag]
