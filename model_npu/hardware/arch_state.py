from typing import Optional

import torch
from ..logging.logger import Logger
from .config import ArchStateConfig


class ArchState:
    def __init__(
        self,
        config: ArchStateConfig,
        logger: Optional[Logger] = None,
    ) -> None:
        self.cfg = config
        self.logger = logger

        self.initialize_buffers()
        self.reset()

    def initialize_buffers(self) -> None:
        self.mem: torch.Tensor = torch.zeros(self.cfg.memory_size, dtype=torch.uint8)
        self.xrf: list[int] = [0] * self.cfg.num_x_registers
        self.mrf: list[torch.Tensor] = [
            torch.zeros(self.cfg.mrf_depth * self.cfg.mrf_width, dtype=torch.uint8)
            for _ in range(self.cfg.num_m_registers)
        ]
        self.wb: dict[str, list[torch.Tensor]] = {}
        self.wb["mxu0"] = [
            torch.zeros(self.cfg.wb_width, dtype=torch.uint8)
            for _ in range(self.cfg.num_wb_registers)
        ]
        self.wb["mxu1"] = [
            torch.zeros(self.cfg.wb_width, dtype=torch.uint8)
            for _ in range(self.cfg.num_wb_registers)
        ]
        self.flags: list[bool] = [False] * 3

    def reset(self) -> None:
        for i in range(len(self.xrf)):
            self.xrf[i] = 0
        for i in range(len(self.mrf)):
            self.mrf[i].fill_(0)
        for i in range(len(self.wb["mxu0"])):
            self.wb["mxu0"][i].fill_(0)
        for i in range(len(self.wb["mxu1"])):
            self.wb["mxu1"][i].fill_(0)
        self.pc = 0
        self.npc = 0

    def set_npc(self, value: int) -> None:
        self.npc = value

    def set_pc(self, value: int) -> None:
        if self.pc == value:
            return
        self.pc = value
        if self.logger:
            self.logger.log_arch_value("pc", 0, value)

    def write_xrf(self, rd: int, value: int) -> None:
        if rd == 0:
            return
        if rd < len(self.xrf) and self.xrf[rd] == value:
            return
        self.xrf[rd] = value
        if self.logger:
            self.logger.log_arch_value("xrf", rd, value)

    def read_xrf(self, rs: int) -> int:
        return self.xrf[rs]

    def write_mrf_u8(self, vd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.uint8
        assert (
            value.numel()
            == self.cfg.mrf_depth * self.cfg.mrf_width // torch.uint8.itemsize
        )
        self.mrf[vd].view(torch.uint8)[:] = value.flatten()

    def read_mrf_u8(self, vs: int) -> torch.Tensor:
        return (
            self.mrf[vs]
            .view(torch.uint8)
            .reshape(self.cfg.mrf_depth, self.cfg.mrf_width // torch.uint8.itemsize)
        )

    def write_mrf_fp8(self, vd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.uint8
        assert (
            value.numel()
            == self.cfg.mrf_depth * self.cfg.mrf_width // torch.float8_e4m3fn.itemsize
        )
        self.mrf[vd].view(torch.float8_e4m3fn)[:] = value.flatten()

    def read_mrf_fp8(self, vs: int) -> torch.Tensor:
        return (
            self.mrf[vs]
            .view(torch.float8_e4m3fn)
            .reshape(
                self.cfg.mrf_depth,
                self.cfg.mrf_width // torch.float8_e4m3fn.itemsize,
            )
        )

    def write_mrf_f32(self, vd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.float32
        assert (
            value.numel()
            == self.cfg.mrf_depth * self.cfg.mrf_width // torch.float32.itemsize
        )
        self.mrf[vd].view(torch.float32)[:] = value.flatten()

    def read_mrf_f32(self, vs: int) -> torch.Tensor:
        return (
            self.mrf[vs]
            .view(torch.float32)
            .reshape(self.cfg.mrf_depth, self.cfg.mrf_width // torch.float32.itemsize)
        )

    def write_mrf_bf16(self, vd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.bfloat16
        assert (
            value.numel()
            == self.cfg.mrf_depth * self.cfg.mrf_width // torch.bfloat16.itemsize
        )
        self.mrf[vd].view(torch.bfloat16)[:] = value.flatten()

    def read_mrf_bf16(self, vs: int) -> torch.Tensor:
        return (
            self.mrf[vs]
            .view(torch.bfloat16)
            .reshape(self.cfg.mrf_depth, self.cfg.mrf_width // torch.bfloat16.itemsize)
        )

    def read_vrf_bf16(self, v: int) -> torch.Tensor:
        vs = v // self.cfg.mrf_depth
        row = v % self.cfg.mrf_depth
        row_bytes = self.cfg.mrf_width  # should be 32
        start = row * row_bytes
        end = start + row_bytes
        row_u8 = self.mrf[vs][start:end].contiguous()
        bf = row_u8.view(torch.int16).view(torch.bfloat16)
        return bf.clone()

    def write_vrf_bf16(self, v: int, value: torch.Tensor) -> None:
        vs = v // self.cfg.mrf_depth
        row = v % self.cfg.mrf_depth
        row_bytes = self.cfg.mrf_width
        start = row * row_bytes
        end = start + row_bytes
        encoded = value.contiguous().view(torch.int16).view(torch.uint8)
        self.mrf[vs][start:end] = encoded

    def write_wb_u8(self, unit: str, wd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.uint8
        assert value.numel() == self.cfg.wb_width // torch.uint8.itemsize
        self.wb[unit][wd].view(torch.uint8)[:] = value.flatten()

    def read_wb_u8(self, unit: str, ws: int) -> torch.Tensor:
        num_rows = self.cfg.mrf_width // torch.uint8.itemsize
        num_cols = (self.cfg.wb_width // torch.uint8.itemsize) // num_rows
        return self.wb[unit][ws].view(torch.uint8).reshape(num_rows, num_cols)

    def write_wb_bf16(self, unit: str, wd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.bfloat16
        assert value.numel() == self.cfg.wb_width // torch.bfloat16.itemsize
        self.wb[unit][wd].view(torch.bfloat16)[:] = value.flatten()

    def read_wb_bf16(self, unit: str, ws: int) -> torch.Tensor:
        num_rows = self.cfg.mrf_width // torch.bfloat16.itemsize
        num_cols = (self.cfg.wb_width // torch.bfloat16.itemsize) // num_rows
        return self.wb[unit][ws].view(torch.bfloat16).reshape(num_rows, num_cols)

    def write_wb_fp8(self, unit: str, wd: int, value: torch.Tensor) -> None:
        assert value.dtype == torch.float8_e4m3fn
        assert value.numel() == self.cfg.wb_width // torch.float8_e4m3fn.itemsize
        self.wb[unit][wd].view(torch.float8_e4m3fn)[:] = value.flatten()

    def read_wb_fp8(self, unit: str, ws: int) -> torch.Tensor:
        num_rows = self.cfg.mrf_width // torch.float8_e4m3fn.itemsize
        num_cols = (self.cfg.wb_width // torch.float8_e4m3fn.itemsize) // num_rows

        return self.wb[unit][ws].view(torch.float8_e4m3fn).reshape(num_rows, num_cols)

    def write_memory(self, base: int, data: torch.Tensor) -> None:
        data = data.flatten()
        print(f"Writing {data.numel()} bytes to memory at base {base}")
        assert (
            base + data.numel() <= self.cfg.memory_size
        ), f"Memory write out of bounds: {base} + {data.numel()} > {self.cfg.memory_size}"
        self.mem[base:base + data.numel()] = data

    def read_memory(self, base: int, length: int) -> torch.Tensor:
        assert (
            base + length <= self.cfg.memory_size
        ), f"Memory read out of bounds: {base} + {length} > {self.cfg.memory_size}"
        return self.mem[base:base + length]

    def set_flag(self, flag: int) -> None:
        self.flags[flag] = True

    def clear_flag(self, flag: int) -> None:
        self.flags[flag] = False

    def check_flag(self, flag: int) -> bool:
        return self.flags[flag]
