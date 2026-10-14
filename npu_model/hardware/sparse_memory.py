"""A byte memory addressed by absolute physical address, materialized by page.

The RTL simulation target (EE290SimConfig) maps DRAM at ``0x8000_0000`` with
``WithExtMemSize(0x10_0000_0000)``, 64 GiB. The model keeps that address map
without allocating it: pages are created on first touch, so an untouched
64 GiB window costs nothing and a program's working set costs what it uses.

Indexing uses absolute addresses, as the TileLink fabric sees them. Slices are
clipped to ``[0, size)`` like a tensor's; the architectural accessors on
``ArchState`` enforce the DRAM window and raise on addresses outside it.
"""
from __future__ import annotations

import torch

_PAGE_BITS = 16
PAGE_BYTES = 1 << _PAGE_BITS


class SparseMemory:
    def __init__(self, size: int, *, seed: int | None = None) -> None:
        if size < 0:
            raise ValueError("memory size cannot be negative")
        self.size = int(size)
        self._seed = seed
        self._pages: dict[int, torch.Tensor] = {}

    # -- tensor-like surface used by the model and tests -------------------

    @property
    def shape(self) -> tuple[int]:
        return (self.size,)

    def numel(self) -> int:
        return self.size

    def __len__(self) -> int:
        return self.size

    @property
    def dtype(self) -> torch.dtype:
        return torch.uint8

    def randomize(self, seed: int) -> None:
        """Make untouched bytes pseudo-random, deterministically per page."""
        self._seed = int(seed)
        self._pages.clear()

    def pages(self) -> int:
        """Pages materialized so far."""
        return len(self._pages)

    def dense(self) -> torch.Tensor:
        """The whole memory as one tensor; for small apertures and tests only."""
        if self.size > (1 << 31):
            raise MemoryError(f"refusing to densify a {self.size} byte memory")
        return self[0:self.size]

    # -- pages -------------------------------------------------------------

    def _page(self, index: int) -> torch.Tensor:
        page = self._pages.get(index)
        if page is None:
            if self._seed is None:
                page = torch.zeros(PAGE_BYTES, dtype=torch.uint8)
            else:
                generator = torch.Generator()
                generator.manual_seed((self._seed * 0x9E3779B97F4A7C15 + index) & 0x7FFF_FFFF_FFFF_FFFF)
                page = torch.randint(0, 256, (PAGE_BYTES,), generator=generator, dtype=torch.uint8)
            self._pages[index] = page
        return page

    def _bounds(self, key) -> tuple[int, int]:
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise IndexError("strided memory slices are not supported")
            start, stop, _ = key.indices(self.size)
            return start, max(start, stop)
        if isinstance(key, int):
            if key < 0:
                key += self.size
            if not 0 <= key < self.size:
                raise IndexError("memory index out of range")
            return key, key + 1
        raise TypeError("memory is indexed by int or slice")

    def __getitem__(self, key) -> torch.Tensor:
        start, stop = self._bounds(key)
        out = torch.empty(stop - start, dtype=torch.uint8)
        address = start
        while address < stop:
            index, offset = address >> _PAGE_BITS, address & (PAGE_BYTES - 1)
            run = min(PAGE_BYTES - offset, stop - address)
            page = self._pages.get(index)
            if page is None and self._seed is None:
                out[address - start:address - start + run].zero_()
            else:
                out[address - start:address - start + run] = self._page(index)[offset:offset + run]
            address += run
        return out

    def __setitem__(self, key, value) -> None:
        start, stop = self._bounds(key)
        length = stop - start
        if isinstance(value, torch.Tensor):
            value = value.flatten()
            if value.numel() != length:
                raise ValueError(f"cannot write {value.numel()} bytes into a {length} byte region")
            if value.dtype != torch.uint8:
                value = value.to(torch.uint8)
        address = start
        while address < stop:
            index, offset = address >> _PAGE_BITS, address & (PAGE_BYTES - 1)
            run = min(PAGE_BYTES - offset, stop - address)
            page = self._page(index)
            if isinstance(value, torch.Tensor):
                page[offset:offset + run] = value[address - start:address - start + run]
            else:
                page[offset:offset + run] = int(value) & 0xFF
            address += run
