"""Atlas geometry and its timing rules

A tensor register holds R rows of W bytes, with R = W = N (the RTL's tile size,
``AtlasParams.withTileDim``). Weight pushes copy one register into a W x W
weight slot, an accumulator holds W BF16 columns. BF16 lanes are W / 2 (one
register row per cycle) and both MXU arrays are W x W. VMEM lines and DMA beats
do not scale: they stay ``vmem_line_bytes`` (32 B), and the LSU converts
between register rows and lines. Every unit moves one register row per port per
cycle, so most windows below are R rows long.

Each size-dependent rule names the RTL it follows (sp26-atlas-acc
``parameterizable``, 6b48d36).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .hardware.config import ArchStateConfig, HardwareConfig


# ---------------------------------------------------------------------------
# VPU operation classes (VectorFSM)
# ---------------------------------------------------------------------------

VPU_TWO_INPUT = frozenset({"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"})
VPU_ROW_REDUCE = frozenset({"vredsum.row.bf16", "vredmin.row.bf16", "vredmax.row.bf16"})
VPU_COL_REDUCE = frozenset({"vredsum.bf16", "vredmin.bf16", "vredmax.bf16"})
VPU_VLI = frozenset({"vli.all", "vli.row", "vli.col", "vli.one"})
VPU_SINGLE = frozenset({"vmov", "vrecip.bf16", "vexp.bf16", "vexp2.bf16", "vrelu.bf16", "vsin.bf16",
                        "vcos.bf16", "vtanh.bf16", "vlog2.bf16", "vsqrt.bf16", "vsquare.bf16",
                        "vcube.bf16", "vpack.bf16.fp8", "vunpack.fp8.bf16"})
VPU_OPS = VPU_TWO_INPUT | VPU_ROW_REDUCE | VPU_COL_REDUCE | VPU_VLI | VPU_SINGLE
VPU_GROUPS = (
    frozenset({"vadd.bf16", "vsub.bf16", "vredsum.row.bf16"}),
    frozenset({"vexp.bf16", "vexp2.bf16"}),
    frozenset({"vsin.bf16", "vcos.bf16"}),
    frozenset({"vsquare.bf16", "vcube.bf16"}),
    frozenset({"vmaximum.bf16", "vredmax.bf16"}),
    frozenset({"vminimum.bf16", "vredmin.bf16"}),
    VPU_VLI,
)
"""Ops sharing lane-box logic: VectorFSM never overlaps two from one group."""
VPU_ISSUE_SLOTS = 2
"""Software-scheduled issue slots; two-input instructions and row reductions take both."""

MXU_MNEMONICS = tuple(
    f"{op}.mxu{unit}" for unit in (0, 1)
    for op in ("vmatpush.weight", "vmatpush.acc.fp8", "vmatpush.acc.bf16",
               "vmatmul.acc", "vmatmul", "vmatpop.fp8.acc", "vmatpop.bf16.acc"))
LSU_SCALAR_LOADS = frozenset({"lb", "lh", "lw", "lbu", "lhu", "seld"})
LSU_SCALAR_STORES = frozenset({"sb", "sh", "sw"})


@dataclass(frozen=True)
class Stream:
    """One tensor register streamed a row every ``stride`` cycles, from issue offset ``first``.

    ``operand`` names the instruction field ("vs1", "vs2" or "vd") and ``offset``
    the register after it (1 for the high half of a BF16 pair). A read stream
    with ``sampled`` False is a second pass whose data the result does not use.
    """

    operand: str
    offset: int
    first: int
    count: int
    stride: int = 1
    sampled: bool = True

    @property
    def last(self) -> int:
        return self.first + self.stride * (self.count - 1)

    def row_at(self, age: int) -> int | None:
        """The row this stream moves at issue offset ``age``, if any."""
        step, remainder = divmod(age - self.first, self.stride)
        return step if remainder == 0 and 0 <= step < self.count else None


@dataclass(frozen=True)
class Geometry:
    rows: int = 32
    """R: rows per tensor register (``mrf_depth``)."""
    row_bytes: int = 32
    """W: bytes per row (``mrf_width``); FP8 columns, MXU inner and output size."""
    num_m_registers: int = 64
    vmem_size: int = 1536 * 1024
    vmem_line_bytes: int = 32
    """A VMEM line: the unit of every VMEM access (``VmemParams.lineBytes``)."""
    vmem_bank_bytes: int = 256 * 1024
    dma_beat_bytes: int = 32

    MXU_INFLIGHT_DEPTH: ClassVar[dict[int, int]] = {0: 3, 1: 2}
    """Sequencer compute trackers; RTL constants, not sizes."""
    MXU1_IPT_STAGES: ClassVar[int] = 2
    """Inner-product tree pipeline stages, ``InnerProductTreeParams.latency``
    (``pipelineCuts = {1}``); independent of the tile size."""

    @classmethod
    def of(cls, arch: ArchStateConfig, hardware: HardwareConfig | None = None) -> Geometry:
        """The geometry of ``arch`` (the architectural state actually built) on ``hardware``."""
        extra = {} if hardware is None else {
            "vmem_bank_bytes": hardware.vmem_bank_bytes, "dma_beat_bytes": hardware.dma_beat_bytes}
        return cls(rows=arch.mrf_depth, row_bytes=arch.mrf_width, num_m_registers=arch.num_m_registers,
                   vmem_size=arch.vmem_size, vmem_line_bytes=arch.vmem_line_bytes, **extra)

    @classmethod
    def from_config(cls, hardware: HardwareConfig) -> Geometry:
        return cls.of(hardware.arch_state_config, hardware)

    # ------------------------------------------------------------------
    # Derived sizes
    # ------------------------------------------------------------------

    @property
    def tile_bytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def lanes_bf16(self) -> int:
        return self.row_bytes // 2

    @property
    def mxu_k(self) -> int:
        """MXU inner dimension: FP8 columns of a register row."""
        return self.row_bytes

    @property
    def mxu_n(self) -> int:
        """MXU output columns: BF16 columns of an accumulator."""
        return self.row_bytes

    @property
    def mreg_banks(self) -> int:
        """Physical 1R1W banks in MregFile.scala: mN and m(N + banks) share one."""
        return self.num_m_registers // 2

    def mreg_bank(self, reg: int) -> int:
        return reg % self.mreg_banks

    def physical_row(self, reg: int, row: int) -> int:
        """Row of ``reg``'s physical bank holding its ``row``."""
        return (reg // self.mreg_banks) * self.rows + row

    # ------------------------------------------------------------------
    # VPU (VectorFSM): one register row per read port per cycle
    # ------------------------------------------------------------------

    def vpu_reads(self, name: str) -> tuple[Stream, ...]:
        r = self.rows
        if name in VPU_VLI:
            return ()
        if name in VPU_ROW_REDUCE:
            # A full BF16 row (both halves of the pair) per cycle.
            return Stream("vs1", 0, 0, r), Stream("vs1", 1, 0, r)
        if name == "vunpack.fp8.bf16":
            return (Stream("vs2", 0, 0, r),)
        source = "vs2" if name == "vpack.bf16.fp8" else "vs1"
        pair = (Stream(source, 0, 0, r), Stream(source, 1, r, r))
        if name in VPU_COL_REDUCE:
            # A second pass over the pair whose data the result does not use.
            return pair + (Stream(source, 0, 2 * r, r, sampled=False),
                           Stream(source, 1, 3 * r, r, sampled=False))
        if name in VPU_TWO_INPUT:
            return pair + (Stream("vs2", 0, 0, r), Stream("vs2", 1, r, r))
        return pair

    def vpu_writes(self, name: str) -> tuple[Stream, ...]:
        r = self.rows
        if name in VPU_ROW_REDUCE:
            # Both halves of row k land together, after the reduction tree.
            first = self.vpu_row_sum_first if name == "vredsum.row.bf16" else 2
            return Stream("vd", 0, first, r), Stream("vd", 1, first, r)
        if name == "vpack.bf16.fp8":
            # FP8 row k packs BF16 rows 2k and 2k + 1, two cycles after the second.
            return (Stream("vd", 0, 3, r, stride=2),)
        if name in {"vli.col", "vli.one"}:
            return (Stream("vd", 0, 1, r),)
        first = (1 if name in VPU_VLI else 3 if name == "vunpack.fp8.bf16"
                 else 2 * r + 2 if name in VPU_COL_REDUCE else 2)
        return Stream("vd", 0, first, r), Stream("vd", 1, first + r, r)

    @property
    def vpu_row_sum_first(self) -> int:
        """First write of ``vredsum.row``: ``AddSubSumVec`` reduces through
        log2(lanes) pipelined adder levels, three cycles after the read (7 at W = 32)."""
        return 3 + (self.lanes_bf16 - 1).bit_length()

    def vpu_read_last(self, name: str) -> int:
        """Issue offset of the last read (-1 when the op reads nothing)."""
        return max((s.last for s in self.vpu_reads(name)), default=-1)

    def vpu_latency(self, name: str) -> int:
        return max(s.last for s in self.vpu_writes(name)) + 1

    def vpu_issue_cycles(self, name: str) -> int:
        """Cycles an instruction holds its issue slot, through its last write."""
        return self.vpu_latency(name) - 1

    def vpu_latencies(self) -> dict[str, int]:
        return {name: self.vpu_latency(name) for name in sorted(VPU_OPS)}

    # ------------------------------------------------------------------
    # XLU (XluEngine): read the register, turn around, write it back
    # ------------------------------------------------------------------

    @property
    def xlu_read_first(self) -> int:
        return 1

    @property
    def xlu_write_first(self) -> int:
        """Reads at 1..R, the last response turns the engine around at R + 1."""
        return self.rows + 2

    @property
    def transpose_cycles(self) -> int:
        """Cycles a transpose holds the engine, from issue through its last row write."""
        return self.xlu_write_first + self.rows

    # ------------------------------------------------------------------
    # MXUs: sequencer ports stream one register row per cycle
    # ------------------------------------------------------------------

    @property
    def mxu_port_cycles(self) -> int:
        """Cycles a command holds each sequencer port it takes."""
        return self.rows

    def mxu_compute_first(self, unit: int) -> int:
        """Issue offset at which a matmul's accumulator row 0 lands.

        MXU0: one cycle, then ``SystolicArray``'s ``rows + cols - 2`` cycles of
        skew and de-skew (PE(i, j) evaluates row r at T + 1 + r + i + j). MXU1:
        one cycle, then the inner-product tree's pipeline stages."""
        return self.mxu_k + self.mxu_n - 1 if unit == 0 else 1 + self.MXU1_IPT_STAGES

    @property
    def mxu0_wslot_drain(self) -> int:
        """A weight push may not reach a slot a matmul issued this many cycles ago
        or fewer still uses (SystolicArraySequencer ``wbufDrainInit = rows + cols - 2``)."""
        return self.mxu_k + self.mxu_n - 2

    @property
    def mxu1_acc_push_gate(self) -> int:
        """MXU1 refuses a matmul or FP8 push on an accumulator a push issued this
        many cycles ago or fewer is writing."""
        return self.rows

    @property
    def mxu_reservation_free(self) -> int:
        """Issue offset from which an MXU command's MREG reservation is gone."""
        return self.rows + 1

    def mxu_latency(self, mnemonic: str) -> int:
        if mnemonic.startswith("vmatmul"):
            return self.mxu_compute_first(int(mnemonic[-1])) + self.rows
        return self.rows + 1

    def mxu_latencies(self) -> dict[str, int]:
        return {name: self.mxu_latency(name) for name in MXU_MNEMONICS}

    # ------------------------------------------------------------------
    # LSU: VLOAD/VSTORE step one beat per cycle, the narrower of a VMEM line
    # and a register row (LSU.scala's WidthConverter). Beat b reads its
    # source at T + 1 + b and writes at T + 3 + b; a source wider than a
    # beat is read on its first beat, a wider destination written on its last.
    # ------------------------------------------------------------------

    @property
    def vls_beats(self) -> int:
        return self.tile_bytes // min(self.vmem_line_bytes, self.row_bytes)

    @property
    def _vls_ratio(self) -> int:
        line, row = self.vmem_line_bytes, self.row_bytes
        return max(line, row) // min(line, row)

    def vls_mreg_stream(self, mnemonic: str) -> Stream:
        """Register rows a VLOAD writes or a VSTORE reads."""
        stride = self._vls_ratio if self.row_bytes > self.vmem_line_bytes else 1
        first = 3 + stride - 1 if mnemonic == "vload" else 1
        return Stream("vd", 0, first, self.rows, stride)

    def vls_line_stream(self, mnemonic: str) -> Stream:
        """VMEM lines (counted from the base) a VLOAD reads or a VSTORE writes."""
        stride = self._vls_ratio if self.vmem_line_bytes > self.row_bytes else 1
        first = 1 if mnemonic == "vload" else 3 + stride - 1
        return Stream("vmem", 0, first, self.tile_bytes // self.vmem_line_bytes, stride)

    @property
    def vls_latency(self) -> int:
        """VLOAD/VSTORE: the last of ``vls_beats`` beats writes at T + beats + 2."""
        return self.vls_beats + 3

    def lsu_latency(self, mnemonic: str) -> int:
        """Inclusive execute-stage cycles, including the scalar command/response flops."""
        if mnemonic in {"vload", "vstore"}:
            return self.vls_latency
        return 4 if mnemonic in LSU_SCALAR_LOADS else 2

    def lsu_latencies(self) -> dict[str, int]:
        return {name: self.lsu_latency(name)
                for name in sorted(LSU_SCALAR_LOADS | LSU_SCALAR_STORES | {"vload", "vstore"})}

    # ------------------------------------------------------------------
    # VMEM addressing
    # ------------------------------------------------------------------

    def vls_byte_address(self, words: int) -> int:
        """VMEM byte address of the line a VLS word operand names."""
        return line_address(words, self.vmem_size, self.vmem_line_bytes)

    def scalar_byte_address(self, address: int) -> int:
        """Scalar LSU byte address: only the VMEM-local address bits reach the LSU."""
        return address & ((1 << (self.vmem_size - 1).bit_length()) - 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def problems(self) -> list[str]:
        """The RTL's ``AtlasParams`` and ``LSU`` requirements this geometry breaks."""
        r, w, line = self.rows, self.row_bytes, self.vmem_line_bytes
        pow2 = lambda x: x > 0 and not x & (x - 1)
        found = []
        if r != w:
            found.append(f"tensor registers must be square: {r} rows of {w} bytes")
        if w < 8 or not pow2(w):
            found.append(f"tile size {w} must be a power of two of at least 8")
        if self.num_m_registers % 2 or not 2 <= self.num_m_registers <= 64:
            found.append(f"{self.num_m_registers} tensor registers: need an even count up to 64 "
                         "(6-bit register fields, banks shared by mN and m(N + count/2))")
        if line < 4 or not pow2(line):
            found.append(f"VMEM line ({line} B) must be a power of two of at least 4 bytes")
        elif self.tile_bytes % line or (w % line and line % w):
            found.append(f"a register row ({w} B) and a VMEM line ({line} B) must divide one another")
        if self.tile_bytes > self.vmem_bank_bytes:
            found.append(f"one register ({self.tile_bytes} B) must fit in a VMEM bank ({self.vmem_bank_bytes} B)")
        if self.dma_beat_bytes != line:
            found.append(f"DMA beat ({self.dma_beat_bytes} B) must equal the VMEM line ({line} B): "
                         "the engine moves one line per beat")
        return found


DEFAULT = Geometry()
"""The 32 x 32 geometry of ``DefaultHardwareConfig``."""


def line_address(words: int, vmem_size: int, line_bytes: int) -> int:
    """VMEM byte address of the line a word operand names.

    ScalarCore and AtlasCore drop the word-offset bits inside a line and keep
    only as many line-address bits as the VMEM has, so higher bits wrap."""
    offset_bits = (line_bytes // 4).bit_length() - 1
    line_bits = (vmem_size // line_bytes - 1).bit_length()
    return ((words >> offset_bits) & ((1 << line_bits) - 1)) * line_bytes


def validate(hardware: HardwareConfig) -> Geometry:
    """The geometry of ``hardware``, or ValueError naming every inconsistency."""
    arch = hardware.arch_state_config
    geometry = Geometry.from_config(hardware)
    found = geometry.problems()
    if arch.wb_width != geometry.tile_bytes:
        found.append(f"weight slot ({arch.wb_width} B) must hold one register ({geometry.tile_bytes} B)")
    if hardware.dma_num_channels > 8:
        found.append(f"{hardware.dma_num_channels} DMA channels: the channel field is 3 bits")
    if found:
        raise ValueError(f"{type(hardware).__name__}: " + "; ".join(found))
    return geometry
