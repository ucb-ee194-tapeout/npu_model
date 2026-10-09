"""Unbounded functional architectural state for ``.fs`` programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import torch

from npu_model.configs.numerics import NUMERICS
from npu_model.hardware.config import ArchStateConfig

T = TypeVar("T")


class VirtualFile(Generic[T]):
    """Sparse, lazily initialized register/bank file."""

    def __init__(self, factory: Callable[[], T]):
        self._factory = factory
        self._values: dict[int, T] = {}

    def __getitem__(self, index: int) -> T:
        if index not in self._values:
            self._values[index] = self._factory()
        return self._values[index]

    def __setitem__(self, index: int, value: T) -> None:
        if isinstance(value, torch.Tensor):
            value = value.clone()  # type: ignore[assignment]
        self._values[index] = value

    def items(self):
        return self._values.items()


class SparseByteMemory:
    """Page-sparse byte memory with zero-filled, byte-addressable semantics."""

    PAGE_BYTES = 4096

    def __init__(self, size: int):
        self.size = size
        self._pages: dict[int, bytearray] = {}

    def __getitem__(self, key: slice) -> torch.Tensor:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Functional memory supports contiguous slices only")
        start = 0 if key.start is None else key.start
        stop = self.size if key.stop is None else key.stop
        if start < 0 or stop < start or stop > self.size:
            raise AssertionError(f"Memory read out of bounds: [{start}, {stop})/{self.size}")
        if start == stop:
            return torch.empty(0, dtype=torch.uint8)
        output = bytearray(stop - start)
        cursor = start
        while cursor < stop:
            page_index, page_offset = divmod(cursor, self.PAGE_BYTES)
            count = min(stop - cursor, self.PAGE_BYTES - page_offset)
            page = self._pages.get(page_index)
            if page is not None:
                output[cursor - start:cursor - start + count] = page[page_offset:page_offset + count]
            cursor += count
        return torch.frombuffer(output, dtype=torch.uint8).clone()

    def __setitem__(self, key: slice, value: torch.Tensor) -> None:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("Functional memory supports contiguous slices only")
        start = 0 if key.start is None else key.start
        data = value.detach().to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes()
        stop = start + len(data) if key.stop is None else key.stop
        if start < 0 or stop < start or stop > self.size or stop - start != len(data):
            raise AssertionError(f"Memory write out of bounds: [{start}, {stop})/{self.size}")
        cursor = start
        data_offset = 0
        while cursor < stop:
            page_index, page_offset = divmod(cursor, self.PAGE_BYTES)
            count = min(stop - cursor, self.PAGE_BYTES - page_offset)
            chunk = data[data_offset:data_offset + count]
            page = self._pages.get(page_index)
            if page is None and any(chunk):
                page = bytearray(self.PAGE_BYTES)
                self._pages[page_index] = page
            if page is not None:
                page[page_offset:page_offset + count] = chunk
                if not any(page):
                    del self._pages[page_index]
            cursor += count
            data_offset += count

    def load(self, offset: int, data: torch.Tensor | bytes | bytearray) -> None:
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(list(data), dtype=torch.uint8)
        self[offset:offset + data.numel()] = data


@dataclass
class FunctionalConfig:
    mrf_depth: int = 32
    mrf_width: int = 32
    wb_width: int = 1024
    num_x_registers: int = 2**63
    num_csrs: int = 4096
    num_e_registers: int = 2**63
    num_m_registers: int = 2**63
    num_wb_registers: int = 2**63
    dram_size: int = 1 * 1024 * 1024 * 1024
    vmem_size: int = 1536 * 1024
    numerics: str = "rtl"
    randomize_init: bool = False
    init_seed: int = 42


class FunctionalState:
    """Duck-compatible state used by the shared instruction ``exec`` methods.

    Register and bank identifiers are virtual integers assigned by the `.fs`
    parser.  Physical capacities and cycle/conflict tracking are not modeled.
    """

    def __init__(self, config: FunctionalConfig | None = None):
        cfg = config or FunctionalConfig()
        self.cfg = ArchStateConfig(**vars(cfg))
        self.math = NUMERICS[cfg.numerics]
        self.xrf: VirtualFile[int] = VirtualFile(lambda: 0)
        self.erf: VirtualFile[int] = VirtualFile(lambda: 0)
        tile_bytes = cfg.mrf_depth * cfg.mrf_width
        self.mrf: VirtualFile[torch.Tensor] = VirtualFile(
            lambda: torch.zeros(tile_bytes, dtype=torch.uint8)
        )
        wb_bytes = cfg.wb_width
        self.wb: dict[str, VirtualFile[torch.Tensor]] = {
            "mxu0": VirtualFile(lambda: torch.zeros(wb_bytes, dtype=torch.uint8)),
            "mxu1": VirtualFile(lambda: torch.zeros(wb_bytes, dtype=torch.uint8)),
        }
        acc_cols = cfg.mrf_width // torch.bfloat16.itemsize * 2
        self.acc: dict[str, VirtualFile[torch.Tensor]] = {
            "mxu0": VirtualFile(lambda: torch.zeros((cfg.mrf_depth, acc_cols), dtype=torch.bfloat16)),
            "mxu1": VirtualFile(lambda: torch.zeros((cfg.mrf_depth, acc_cols), dtype=torch.bfloat16)),
        }
        self.dram = SparseByteMemory(cfg.dram_size)
        self.vmem = SparseByteMemory(cfg.vmem_size)
        self.base = 0
        self.execute_pc = 0
        self.npc = 0
        self.redirect_requested = False
        self.halted = False
        self.halt_reason: str | None = None
        self._csr_values = {addr: 0 for addr in (0xC00, 0xC01, 0xC02, 0xC03, 0xC10, 0xC11)}
        self._csr_written: set[int] = set()

    def reset(self) -> None:
        self.xrf = VirtualFile(lambda: 0)
        self.erf = VirtualFile(lambda: 0)
        self.mrf = VirtualFile(lambda: torch.zeros(self.cfg.mrf_depth * self.cfg.mrf_width, dtype=torch.uint8))
        self.wb = {
            unit: VirtualFile(lambda: torch.zeros(self.cfg.wb_width, dtype=torch.uint8))
            for unit in ("mxu0", "mxu1")
        }
        acc_cols = self.cfg.mrf_width // torch.bfloat16.itemsize * 2
        self.acc = {
            unit: VirtualFile(lambda: torch.zeros((self.cfg.mrf_depth, acc_cols), dtype=torch.bfloat16))
            for unit in ("mxu0", "mxu1")
        }
        self.dram = SparseByteMemory(self.cfg.dram_size)
        self.vmem = SparseByteMemory(self.cfg.vmem_size)
        self.base = 0
        self.execute_pc = self.npc = 0
        self.redirect_requested = self.halted = False
        self.halt_reason = None
        self._csr_values = {addr: 0 for addr in (0xC00, 0xC01, 0xC02, 0xC03, 0xC10, 0xC11)}
        self._csr_written.clear()

    def set_npc(self, value: int) -> None:
        self.npc = value & 0xFFFFFFFF
        self.redirect_requested = True

    def read_xrf(self, rs: int) -> int:
        return self.xrf[rs]

    def write_xrf(self, rd: int, value: int) -> None:
        # Unlike hardware x0, the functional file contains only named virtuals.
        self.xrf[rd] = value & 0xFFFFFFFF

    def read_erf(self, rs: int) -> int:
        return self.erf[rs]

    def write_erf(self, rd: int, value: int) -> None:
        self.erf[rd] = value & 0xFF

    def write_mrf_u8(self, vd: int, value: torch.Tensor) -> None:
        assert value.numel() == self.cfg.mrf_depth * self.cfg.mrf_width
        self.mrf[vd] = value.to(torch.uint8).flatten()

    def read_mrf_u8(self, vs: int) -> torch.Tensor:
        return self.mrf[vs].reshape(self.cfg.mrf_depth, self.cfg.mrf_width)

    def write_mrf_fp8(self, vd: int, value: torch.Tensor) -> None:
        self.write_mrf_u8(vd, value.contiguous().view(torch.uint8))

    def read_mrf_fp8(self, vs: int) -> torch.Tensor:
        return self.read_mrf_u8(vs).view(torch.float8_e4m3fn)

    def write_mrf_bf16(self, vd: int, value: torch.Tensor) -> None:
        self.write_mrf_u8(vd, value.contiguous().view(torch.uint8))

    def read_mrf_bf16(self, vs: int) -> torch.Tensor:
        return self.read_mrf_u8(vs).view(torch.bfloat16)

    def read_mrf_f32(self, vs: int) -> torch.Tensor:
        return self.read_mrf_u8(vs).view(torch.float32)

    def write_mrf_f32(self, vd: int, value: torch.Tensor) -> None:
        self.write_mrf_u8(vd, value.contiguous().view(torch.uint8))

    def write_mrf_bf16_tile(self, vd: int, value: torch.Tensor) -> None:
        cols = self.cfg.mrf_width // torch.bfloat16.itemsize
        self.write_mrf_bf16(vd, value[:, :cols].contiguous())
        self.write_mrf_bf16(vd + 1, value[:, cols:].contiguous())

    def read_mrf_bf16_tile(self, vs: int) -> torch.Tensor:
        return torch.cat((self.read_mrf_bf16(vs), self.read_mrf_bf16(vs + 1)), dim=1)

    def read_mrf_bf16_transposed(self, vs: int) -> torch.Tensor:
        cols = self.cfg.mrf_width // torch.bfloat16.itemsize
        return self.mrf[vs].view(torch.bfloat16).reshape(cols, self.cfg.mrf_depth)

    def read_vrf_bf16(self, index: int) -> torch.Tensor:
        reg, row = divmod(index, self.cfg.mrf_depth)
        start = row * self.cfg.mrf_width
        return self.mrf[reg][start:start + self.cfg.mrf_width].view(torch.bfloat16).clone()

    def write_vrf_bf16(self, index: int, value: torch.Tensor) -> None:
        reg, row = divmod(index, self.cfg.mrf_depth)
        start = row * self.cfg.mrf_width
        encoded = value.contiguous().view(torch.uint8)
        self.mrf[reg][start:start + self.cfg.mrf_width] = encoded

    def write_wb_u8(self, unit: str, wd: int, value: torch.Tensor) -> None:
        self.wb[unit][wd] = value.to(torch.uint8).flatten()

    def read_wb_u8(self, unit: str, ws: int) -> torch.Tensor:
        rows = self.cfg.mrf_width
        return self.wb[unit][ws].reshape(rows, self.cfg.wb_width // rows)

    def write_wb_bf16(self, unit: str, wd: int, value: torch.Tensor) -> None:
        self.write_wb_u8(unit, wd, value.contiguous().view(torch.uint8))

    def read_wb_bf16(self, unit: str, ws: int) -> torch.Tensor:
        rows = self.cfg.mrf_width // torch.bfloat16.itemsize
        return self.wb[unit][ws].view(torch.bfloat16).reshape(rows, (self.cfg.wb_width // 2) // rows)

    def write_wb_fp8(self, unit: str, wd: int, value: torch.Tensor) -> None:
        self.write_wb_u8(unit, wd, value.contiguous().view(torch.uint8))

    def read_wb_fp8(self, unit: str, ws: int) -> torch.Tensor:
        rows = self.cfg.mrf_width
        return self.wb[unit][ws].view(torch.float8_e4m3fn).reshape(rows, self.cfg.wb_width // rows)

    def write_acc_bf16(self, unit: str, wd: int, value: torch.Tensor) -> None:
        self.acc[unit][wd] = value.to(torch.bfloat16)

    def read_acc_bf16(self, unit: str, ws: int) -> torch.Tensor:
        return self.acc[unit][ws].clone()

    @staticmethod
    def _csr_address(address: int) -> int:
        known = (0xC00, 0xC01, 0xC02, 0xC03, 0xC10, 0xC11)
        return address if address in known else 0xC00

    def read_csrf(self, address: int) -> int:
        address = self._csr_address(address)
        if address == 0xC02:
            reason = {None: 0, "illegal": 1, "ecall": 2, "ebreak": 3}[self.halt_reason]
            return (reason << 1) | int(self.halted)
        return self._csr_values[address]

    def write_csrf(self, address: int, value: int) -> None:
        address = self._csr_address(address)
        if address in (0xC02, 0xC03):
            return
        self._csr_values[address] = value & 0xFFFFFFFF
        self._csr_written.add(address)

    def read_dram(self, offset: int, length: int) -> torch.Tensor:
        address = (self.base << 32) | offset
        if address < 0 or address + length > self.cfg.dram_size:
            raise AssertionError(f"DRAM read out of bounds: [{address}, {address + length})")
        return self.dram[address:address + length]

    def write_dram(self, offset: int, data: torch.Tensor) -> None:
        address = (self.base << 32) | offset
        data = data.flatten().to(torch.uint8)
        if address < 0 or address + data.numel() > self.cfg.dram_size:
            raise AssertionError(f"DRAM write out of bounds: [{address}, {address + data.numel()})")
        self.dram[address:address + data.numel()] = data

    def read_vmem(self, base: int, offset: int, length: int) -> torch.Tensor:
        address = base + offset
        if address < 0 or address + length > self.cfg.vmem_size:
            raise AssertionError(f"VMEM read out of bounds: [{address}, {address + length})")
        return self.vmem[address:address + length]

    def write_vmem(self, base: int, offset: int, data: torch.Tensor) -> None:
        address = base + offset
        data = data.flatten().to(torch.uint8)
        if address < 0 or address + data.numel() > self.cfg.vmem_size:
            raise AssertionError(f"VMEM write out of bounds: [{address}, {address + data.numel()})")
        self.vmem[address:address + data.numel()] = data

    def load_dram(self, offset: int, data: torch.Tensor | bytes | bytearray) -> None:
        self.dram.load(offset, data)

    def load_vmem(self, offset: int, data: torch.Tensor | bytes | bytearray) -> None:
        self.vmem.load(offset, data)
