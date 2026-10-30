"""Issue-timing rules for Atlas instructions, derived from resource occupancy.

The question this answers: when may instruction B issue after instruction A?
There is one rule, applied to every resource the two instructions share:

    B's use of the resource must come after A's.

"After" means one of three things, depending on the resource:

* **Exclusive occupancy** (a register-file port, an MXU read/write port, a
  push write path, the accumulator's single read or write port, a VMEM bank,
  an engine path): the two occupancy intervals may not overlap.
      G >= last_A - first_B + 1
* **Row data flow** on a buffer row (an accumulator row, a weight-slot
  column, a tensor-register row, a VMEM line, a scalar register): a read must
  come strictly after the write it depends on (RAW, read >= write + 1), and a
  write must not come before a read of the old value (WAR, write >= read;
  the same cycle is fine because every unit samples reads before it applies
  writes).
      G >= max over rows of (producer cycle - consumer offset + 1)
* **Logical register reservation** (ScalarCore's MregBankTracker): B may not
  read or write a tensor register A still has a pending write on, nor write
  one A still reads (VLOAD is exempt from the latter). Each reservation ends
  at a fixed "free from" cycle.
      G >= free_from_A

Two further mechanisms are needed to express the sequencers exactly:

* **Sequencer gates**, one-directional: A raises a gate over a window, and a
  later command that probes it may not issue while it is raised. These are
  the decisions that are not plain data flow (MXU accumulator row-0 gate,
  MXU0 weight-slot drain, MXU1 push gate, the VPU issue-busy bits, the LSU
  scalar writeback slot).
* **Counted slots**: the MXU in-flight trackers (3 on MXU0, 2 on MXU1) and
  the two VPU slots.

Everything instruction-specific is a *table*: ``profile()`` says, for each
operation, which resources it holds for which issue-relative cycles, which
rows it reads and writes when, and what it reserves. Those windows are
transcribed from the RTL sequencers (SystolicArraySequencer.scala,
InnerProductTreesSequencer.scala, VectorFSM, XLU, LSU) and from the model in
``npu_model/hardware``; ``scripts/sweep_issue_rules.py`` checks that the
rules reproduce the model's accept/reject decision for every pair and gap.

The issue gap G is B's issue cycle minus A's. ``DELAY N`` between A and B
gives G = N + 2; G = 1 needs no DELAY.

DMA transfers are the one thing without a fixed window: a channel stays busy
until the memory system finishes the transfer, so the only rule is that a
second transfer on the same channel needs a ``dma.wait`` in between. The
engine reports that, and treats every issue cycle after a ``dma.wait`` as
relative to that wait's release.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

ROWS = 32
LINE_BYTES = 32
VMEM_BANK_BYTES = 256 * 1024
MXU_COMPUTE_FIRST = {0: 63, 1: 3}     # issue offset at which result row 0 lands
MXU_INFLIGHT_DEPTH = {0: 3, 1: 2}     # sequencer in-flight tracker depth
MXU0_WSLOT_DRAIN = 62                 # SystolicArraySequencer.wbufDrainInit
MXU1_ACC_PUSH_GATE = 32               # mxu.py: acc_pushes use `age <= 32`
MXU_PORT_FREE = 33                    # MREG reservation free-from for all MXU ops
FOREVER = 1 << 30

Resource = tuple


@dataclass(frozen=True)
class Hold:
    """Exclusive occupancy of ``resource`` over issue-relative cycles [first, last]."""

    resource: Resource
    first: int
    last: int
    phase: int | None = None
    """VectorFSM read-port holds carry the issue offset of row 0: two VPU ops whose
    streams over one bank are in phase (same row, same cycle) share one SRAM read."""


@dataclass(frozen=True)
class RowAccess:
    """A read or write of one row of ``resource`` at issue-relative cycle ``at``."""

    resource: Resource
    row: int
    at: int
    write: bool


@dataclass(frozen=True)
class Gate:
    """A sequencer gate A raises over [first, last]: a later command that probes
    ``resource`` may not issue while it is raised (one-directional, unlike Hold)."""

    resource: Resource
    first: int
    last: int


@dataclass(frozen=True)
class Probe:
    """B's issue-relative cycle at which it tests a sequencer gate."""

    resource: Resource
    at: int = 0


@dataclass(frozen=True)
class Counted:
    """Occupancy of one of ``capacity`` identical slots over [first, last]."""

    resource: Resource
    first: int
    last: int
    capacity: int


@dataclass
class Profile:
    name: str
    holds: list[Hold] = field(default_factory=list)
    rows: list[RowAccess] = field(default_factory=list)
    counted: list[Counted] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    probes: list[Probe] = field(default_factory=list)
    reads: frozenset[int] = frozenset()      # logical MREG reservation
    writes: frozenset[int] = frozenset()
    reads_free: int = 0                      # issue offset from which the reservation is gone
    writes_free: int = 0
    writer_ignores_readers: bool = False     # VLOAD checks pending writes only


# ---------------------------------------------------------------------------
# Instruction descriptors
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
VPU_LATENCY = {
    **dict.fromkeys(VPU_TWO_INPUT | VPU_SINGLE, 66),
    **dict.fromkeys(VPU_COL_REDUCE, 130),
    "vredsum.row.bf16": 39, "vredmin.row.bf16": 34, "vredmax.row.bf16": 34,
    "vli.all": 65, "vli.row": 65, "vli.col": 33, "vli.one": 33,
    "vunpack.fp8.bf16": 67,
}
SCALAR_LOADS = frozenset({"lb", "lh", "lw", "lbu", "lhu", "seld"})
SCALAR_STORES = frozenset({"sb", "sh", "sw"})
BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu"})
MXU_KINDS = ("weight", "push_fp8", "push_bf16", "pop_fp8", "pop_bf16", "matmul", "matmul_acc")


@dataclass(frozen=True)
class Op:
    """Unit-level description of an instruction (what ``profile`` consumes).

    kind: one of MXU_KINDS | "vpu" (with ``name``) | "xlu" | "vload" | "vstore"
          | "sload" | "sstore" | "scalar" | "seli" | "branch" | "dma" | "dma_wait" | "other"
    """

    kind: str
    mxu: int | None = None
    name: str | None = None       # VPU mnemonic
    mreg: int | None = None       # vs1 / source (MXU, VPU, XLU, vstore) or vd (pops, vload)
    mreg2: int | None = None      # vs2 of a two-input VPU op
    mdst: int | None = None       # vd of VPU / XLU ops
    acc: int | None = None
    wslot: int | None = None
    push_port: int | None = None  # set by the scheduler: 1 if ReadP1 is free at issue, else 0
    rd: int | None = None         # scalar destination (sload, scalar)
    rs: tuple[int, ...] = ()      # scalar sources read at issue
    vmem_addr: int | None = None  # byte address of an LSU VMEM access
    channel: int | None = None    # DMA channel

    @property
    def label(self) -> str:
        if self.kind == "vpu":
            return self.name or "vpu"
        unit = f".mxu{self.mxu}" if self.mxu is not None else ""
        chan = f".ch{self.channel}" if self.channel is not None else ""
        return f"{self.kind}{unit}{chan}"


def op_from_instruction(insn, xrf: Sequence[int] | None = None, push_port: int | None = None) -> Op:
    """Build an ``Op`` from an npu_model Instruction. ``xrf`` (scalar register values)
    is needed to resolve LSU VMEM addresses; without it they are left unknown."""
    name = insn.mnemonic
    if name.startswith("vmat"):
        mxu = int(name[-1])
        if name.startswith("vmatpush.weight"):
            return Op("weight", mxu, mreg=int(insn.vs1), wslot=int(insn.vd) & 1, push_port=push_port)
        if name.startswith("vmatpush.acc"):
            kind = "push_bf16" if "bf16" in name else "push_fp8"
            return Op(kind, mxu, mreg=int(insn.vs1), acc=int(insn.vd) & 1, push_port=push_port)
        if name.startswith("vmatpop"):
            kind = "pop_bf16" if "bf16" in name else "pop_fp8"
            return Op(kind, mxu, mreg=int(insn.vd), acc=int(insn.vs2) & 1)
        kind = "matmul_acc" if ".acc." in name else "matmul"
        return Op(kind, mxu, mreg=int(insn.vs1), acc=int(insn.vd) & 1, wslot=int(insn.vs2) & 1)
    if name == "vtrpose.xlu":
        return Op("xlu", mreg=int(insn.vs1), mdst=int(insn.vd))
    if name in {"vload", "vstore"}:
        addr = None
        if xrf is not None:
            imm = int(insn.imm) & 0xFFF
            imm = imm - 0x1000 if imm & 0x800 else imm
            addr = (((xrf[int(insn.rs1)] + (imm << 5)) >> 3) & 0xFFFF) * 32
        return Op(name, mreg=int(insn.vd), rs=(int(insn.rs1),), vmem_addr=addr)
    if name in VPU_OPS:
        if name in VPU_VLI:
            return Op("vpu", name=name, mdst=int(insn.vd))
        src = int(insn.vs2 if name in {"vpack.bf16.fp8", "vunpack.fp8.bf16"} else insn.vs1)
        src2 = int(insn.vs2) if name in VPU_TWO_INPUT else None
        return Op("vpu", name=name, mreg=src, mreg2=src2, mdst=int(insn.vd))
    if name in SCALAR_LOADS or name in SCALAR_STORES:
        addr = None
        if xrf is not None:
            imm = int(insn.imm) & 0xFFF
            imm = imm - 0x1000 if imm & 0x800 else imm
            addr = (xrf[int(insn.rs1)] + imm) & (VMEM_BANK_BYTES * 8 - 1)
        if name in SCALAR_LOADS:
            return Op("sload", rd=None if name == "seld" else int(insn.rd), rs=(int(insn.rs1),), vmem_addr=addr)
        return Op("sstore", rs=(int(insn.rs1), int(insn.rs2)), vmem_addr=addr)
    if name.startswith("dma."):
        channel = int(getattr(insn, "ch", getattr(insn, "funct3", 0)))
        if name.startswith("dma.wait"):
            return Op("dma_wait", channel=channel)
        if name.startswith("dma.load") or name.startswith("dma.store"):
            rs = tuple(int(getattr(insn, f)) for f in ("rd", "rs1", "rs2") if hasattr(insn, f))
            return Op("dma", channel=channel, rs=rs)
        return Op("scalar", rs=(int(insn.rs1),) if hasattr(insn, "rs1") else ())  # dma.config
    if name == "seli":
        return Op("seli")
    if name == "delay":
        return Op("other")
    if name in {"jal", "jalr"} or name in BRANCHES:
        rs = tuple(int(getattr(insn, f)) for f in ("rs1", "rs2") if hasattr(insn, f))
        rd = int(insn.rd) if hasattr(insn, "rd") else None
        return Op("branch", rd=rd, rs=rs)
    rd = int(insn.rd) if hasattr(insn, "rd") else None
    rs = tuple(int(getattr(insn, f)) for f in ("rs1", "rs2") if hasattr(insn, f))
    return Op("scalar", rd=rd, rs=rs)


# ---------------------------------------------------------------------------
# Access profiles (the table)
# ---------------------------------------------------------------------------

def _mreg_stream(bank: int, first: int, write: bool, rows: Iterable[int] = range(ROWS),
                 stride: int = 1, phase: int | None = None) -> tuple[Hold, list[RowAccess]]:
    """Rows of tensor register ``bank`` from ``first``, one per ``stride`` cycles."""
    rows = list(rows)
    port = Hold(("mrf_port", bank % 32, "w" if write else "r"), first, first + stride * (len(rows) - 1), phase)
    # mN and m(N+32) share a physical 1R1W bank; a read and a write of the same physical row
    # in the same cycle return undefined data (MregFile.scala SyncReadMem).
    accesses = [RowAccess(("mrf", bank), r, first + stride * i, write) for i, r in enumerate(rows)]
    accesses += [RowAccess(("mrf_phys", bank % 32), (bank // 32) * 32 + r, first + stride * i, write)
                 for i, r in enumerate(rows)]
    return port, accesses


def _mxu_profile(op: Op) -> Profile:
    u, kind = op.mxu, op.kind
    p = Profile(op.label)
    acc = ("acc", u, op.acc)
    acc_rd, acc_wr = ("acc_rport", u, op.acc), ("acc_wport", u, op.acc)
    first_row = MXU_COMPUTE_FIRST[u]
    pair = kind in {"push_bf16", "pop_bf16"}
    mregs = [op.mreg, op.mreg + 1] if pair else [op.mreg]

    if kind.startswith("pop"):
        # Reads accumulator rows r at T+r through its one read port; writes vd rows at T+1+r
        # through MXU write ports W0 (and W1 for the BF16 pair).
        p.holds.append(Hold(acc_rd, 0, ROWS - 1))
        p.rows += [RowAccess(acc, r, r, False) for r in range(ROWS)]
        for i, bank in enumerate(mregs):
            p.holds.append(Hold(("mxu_port", u, f"W{i}"), 0, ROWS - 1))
            port, rows = _mreg_stream(bank, 1, True)
            p.holds.append(port)
            p.rows += rows
        p.writes, p.writes_free = frozenset(mregs), MXU_PORT_FREE
        p.probes.append(Probe(("acc_row0_gate", u, op.acc)))
        return p

    # Everything else streams vs1 (and vs1+1) through MXU read ports at T..T+31.
    if kind in {"matmul", "matmul_acc"}:
        ports = ["P0"]
    elif kind == "push_bf16":
        ports = ["P0", "P1"]
    else:
        ports = ["P1" if op.push_port in (None, 1) else "P0"]
    for i, bank in enumerate(mregs):
        p.holds.append(Hold(("mxu_port", u, ports[i]), 0, ROWS - 1))
        port, rows = _mreg_stream(bank, 0, False)
        p.holds.append(port)
        p.rows += rows
    p.reads, p.reads_free = frozenset(mregs), MXU_PORT_FREE

    if kind == "weight":
        # One weight push at a time per MXU; column c lands at T+1+c.
        p.holds.append(Hold(("wpath", u), 1, ROWS))
        p.rows += [RowAccess(("wslot", u, op.wslot), c, 1 + c, True) for c in range(ROWS)]
        if u == 0:
            p.probes.append(Probe(("wslot_drain", 0, op.wslot)))
        else:
            # MXU1 refuses a push to a slot an active (age < 32) push or matmul uses.
            p.gates.append(Gate(("wslot_gate", 1, op.wslot), 0, ROWS - 1))
            p.probes.append(Probe(("wslot_gate", 1, op.wslot)))
        return p

    if kind.startswith("push"):
        # One accumulator push at a time per MXU; row r lands at T+1+r through the
        # accumulator's single write port.
        p.holds.append(Hold(("apath", u), 1, ROWS))
        p.holds.append(Hold(acc_wr, 1, ROWS))
        p.rows += [RowAccess(acc, r, 1 + r, True) for r in range(ROWS)]
        p.probes.append(Probe(("acc_row0_gate", u, op.acc)))
        if u == 1:
            p.gates.append(Gate(("acc_pushgate", 1, op.acc), 0, MXU1_ACC_PUSH_GATE))
            if kind == "push_fp8":
                p.probes.append(Probe(("acc_pushgate", 1, op.acc)))
        return p

    # matmul / matmul_acc
    p.counted.append(Counted(("inflight", u), 0, first_row + ROWS - 1, MXU_INFLIGHT_DEPTH[u]))
    p.holds.append(Hold(acc_wr, first_row, first_row + ROWS - 1))
    p.rows += [RowAccess(acc, r, first_row + r, True) for r in range(ROWS)]
    # The sequencer refuses any command on this accumulator until row 0 has landed.
    p.gates.append(Gate(("acc_row0_gate", u, op.acc), 0, first_row))
    p.probes.append(Probe(("acc_row0_gate", u, op.acc)))
    if kind == "matmul_acc":
        p.holds.append(Hold(acc_rd, 0, ROWS - 1))
        p.rows += [RowAccess(acc, r, r, False) for r in range(ROWS)]
    if u == 0:
        # PE(i, j) uses column j for output row r at T+1+r+i+j: column c is live
        # from T+1+c to T+63+c. The sequencer gate is one cycle more conservative.
        p.rows += [RowAccess(("wslot", 0, op.wslot), c, 1 + c, False) for c in range(ROWS)]
        p.rows += [RowAccess(("wslot", 0, op.wslot), c, 1 + c + 2 * (ROWS - 1), False)
                   for c in range(ROWS)]
        p.gates.append(Gate(("wslot_drain", 0, op.wslot), 0, MXU0_WSLOT_DRAIN))
    else:
        # Each IPT row reads the whole slot: every column at T+1..T+32.
        p.rows += [RowAccess(("wslot", 1, op.wslot), c, t, False)
                   for c in range(ROWS) for t in (1, ROWS)]
        p.gates.append(Gate(("wslot_gate", 1, op.wslot), 0, ROWS - 1))
        p.probes.append(Probe(("wslot_gate", 1, op.wslot)))
        p.probes.append(Probe(("acc_pushgate", 1, op.acc)))
    return p


def _vpu_profile(op: Op) -> Profile:
    """VectorFSM: row schedules from vpu.py, issue-busy bits from ``_issue_blocked``."""
    name = op.name
    p = Profile(name)
    latency = VPU_LATENCY[name]
    double = name in VPU_TWO_INPUT or name in VPU_ROW_REDUCE

    # Reads.
    if name in VPU_VLI:
        sources: list[int] = []
    elif name in VPU_ROW_REDUCE:
        sources = [op.mreg, op.mreg + 1]
        for bank in sources:
            port, rows = _mreg_stream(bank, 0, False, phase=0)
            p.holds.append(port)
            p.rows += rows
    elif name == "vunpack.fp8.bf16":
        sources = [op.mreg]
        port, rows = _mreg_stream(op.mreg, 0, False, phase=0)
        p.holds.append(port)
        p.rows += rows
    else:
        sources = [op.mreg, op.mreg + 1]
        passes = 2 if name in VPU_COL_REDUCE else 1
        for k in range(passes):
            for half in (0, 1):
                port, rows = _mreg_stream(op.mreg + half, 64 * k + 32 * half, False, phase=64 * k + 32 * half)
                p.holds.append(port)
                p.rows += rows
        if name in VPU_TWO_INPUT:
            sources += [op.mreg2, op.mreg2 + 1]
            for half in (0, 1):
                port, rows = _mreg_stream(op.mreg2 + half, 32 * half, False, phase=32 * half)
                p.holds.append(port)
                p.rows += rows
    read_last = (-1 if name in VPU_VLI else 31 if name in VPU_ROW_REDUCE or name == "vunpack.fp8.bf16"
                 else 127 if name in VPU_COL_REDUCE else 63)

    # Writes.
    if name in VPU_ROW_REDUCE:
        first = 7 if name == "vredsum.row.bf16" else 2
        for half in (0, 1):
            port, rows = _mreg_stream(op.mdst + half, first, True)
            p.holds.append(port)
            p.rows += rows
        dests = [op.mdst, op.mdst + 1]
    elif name == "vpack.bf16.fp8":
        port, rows = _mreg_stream(op.mdst, 3, True, stride=2)
        p.holds.append(port)
        p.rows += rows
        dests = [op.mdst]
    else:
        first = 1 if name in VPU_VLI else 3 if name == "vunpack.fp8.bf16" else 66 if name in VPU_COL_REDUCE else 2
        pair = name not in {"vli.col", "vli.one"}
        dests = [op.mdst, op.mdst + 1] if pair else [op.mdst]
        for half, bank in enumerate(dests):
            port, rows = _mreg_stream(bank, first + 32 * half, True)
            p.holds.append(port)
            p.rows += rows
    p.reads, p.writes = frozenset(sources), frozenset(dests)
    p.reads_free, p.writes_free = read_last + 1, latency

    # Issue-busy bits: an op is live while age < latency - 1. B is refused when any live op
    # exists and (both slots are live, or B or the live op is a two-input/row-reduction op,
    # or they share a lane-box logic group or are the same op).
    live_last = latency - 2
    p.counted.append(Counted(("vpu_slot",), 0, live_last, 2))
    p.gates.append(Gate(("vpu_any",), 0, live_last))
    p.probes.append(Probe(("vpu_double",)))
    if double:
        p.gates.append(Gate(("vpu_double",), 0, live_last))
        p.probes.append(Probe(("vpu_any",)))
    p.gates.append(Gate(("vpu_op", name), 0, live_last))
    p.probes.append(Probe(("vpu_op", name)))
    for index, group in enumerate(VPU_GROUPS):
        if name in group:
            p.gates.append(Gate(("vpu_group", index), 0, live_last))
            p.probes.append(Probe(("vpu_group", index)))
    return p


def _xlu_profile(op: Op) -> Profile:
    """vtrpose.xlu: reads vs1 rows at T+1..T+32, writes vd rows at T+34..T+65."""
    p = Profile(op.label)
    port, rows = _mreg_stream(op.mreg, 1, False)
    p.holds.append(port)
    p.rows += rows
    port, rows = _mreg_stream(op.mdst, 34, True)
    p.holds.append(port)
    p.rows += rows
    p.holds.append(Hold(("xlu",), 0, 65))
    p.reads, p.writes = frozenset({op.mreg}), frozenset({op.mdst})
    p.reads_free, p.writes_free = 34, 66
    return p


def _vmem_access(p: Profile, addr: int | None, first: int, count: int, write: bool) -> None:
    """``count`` consecutive 32-byte lines from ``addr``, one per cycle from ``first``; the
    bank (256 KiB) has one port. Unknown addresses contribute nothing."""
    if addr is None:
        return
    p.holds.append(Hold(("vmem_bank", addr // VMEM_BANK_BYTES), first, first + count - 1))
    p.rows += [RowAccess(("vmem",), addr // LINE_BYTES + i, first + i, write) for i in range(count)]


def _lsu_profile(op: Op) -> Profile:
    p = Profile(op.label)
    p.rows += [RowAccess(("xrf", r), 0, 0, False) for r in op.rs if r]
    if op.kind == "vload":
        port, rows = _mreg_stream(op.mreg, 3, True)
        p.holds += [port, Hold(("vload_path",), 0, 34)]
        p.rows += rows
        _vmem_access(p, op.vmem_addr, 1, ROWS, False)
        p.writes, p.writes_free = frozenset({op.mreg}), 35
        p.writer_ignores_readers = True
    elif op.kind == "vstore":
        port, rows = _mreg_stream(op.mreg, 1, False)
        p.holds += [port, Hold(("vstore_path",), 0, 34)]
        p.rows += rows
        _vmem_access(p, op.vmem_addr, 3, ROWS, True)
        p.reads, p.reads_free = frozenset({op.mreg}), 35
    elif op.kind == "sload":
        # One response in flight at a time; the word is read at T+1 and rd written at T+3,
        # when no scalar instruction may be writing a register (no bypass either).
        p.holds.append(Hold(("lsu_scalar",), 0, 2))
        _vmem_access(p, op.vmem_addr, 1, 1, False)
        p.gates.append(Gate(("scalar_writeback",), 3, 3))
        if op.rd:
            p.rows.append(RowAccess(("xrf", op.rd), 0, 3, True))
    else:  # sstore
        _vmem_access(p, op.vmem_addr, 1, 1, True)
    return p


def _scalar_profile(op: Op) -> Profile:
    p = Profile(op.label)
    p.rows += [RowAccess(("xrf", r), 0, 0, False) for r in op.rs if r]
    if op.kind == "seli" or op.rd:
        p.probes.append(Probe(("scalar_writeback",)))
    if op.rd:
        p.rows.append(RowAccess(("xrf", op.rd), 0, 0, True))
    if op.kind == "branch":
        # Taken branches and jumps have one delay slot; a control-flow instruction there is
        # illegal (ScalarCore halts). Branches are treated as taken.
        p.gates.append(Gate(("delay_slot",), 1, 1))
        p.probes.append(Probe(("delay_slot",)))
    return p


def _dma_profile(op: Op) -> Profile:
    p = Profile(op.label)
    if op.kind == "dma":
        p.rows += [RowAccess(("xrf", r), 0, 0, False) for r in op.rs if r]
        p.holds.append(Hold(("dma_channel", op.channel), 0, FOREVER))
    return p


def profile(op: Op) -> Profile:
    """Resource-occupancy profile of one instruction, offsets relative to its issue cycle."""
    if op.mxu is not None:
        return _mxu_profile(op)
    if op.kind == "vpu":
        return _vpu_profile(op)
    if op.kind == "xlu":
        return _xlu_profile(op)
    if op.kind in {"vload", "vstore", "sload", "sstore"}:
        return _lsu_profile(op)
    if op.kind in {"scalar", "seli", "branch"}:
        return _scalar_profile(op)
    if op.kind in {"dma", "dma_wait"}:
        return _dma_profile(op)
    return Profile(op.label)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Violation:
    """Why B cannot issue ``gap`` cycles after A, and the smallest gap that would do."""

    rule: str
    resource: Resource
    earlier: str
    later: str
    gap: int
    required: int
    enforced: bool = True   # False: a value hazard neither the RTL nor the model detects

    def __str__(self) -> str:
        need = "a dma.wait" if self.required >= FOREVER else f"gap >= {self.required}"
        return (f"{self.later} at +{self.gap} after {self.earlier}: {self.rule} on "
                f"{_fmt(self.resource)} needs {need}")


def _fmt(resource: Resource) -> str:
    return ".".join(str(part) for part in resource)


def pair_violations(a: Profile, b: Profile, gap: int) -> list[Violation]:
    """Every rule B (issued ``gap`` cycles after A) breaks against A alone."""
    out: list[Violation] = []

    def add(rule: str, resource: Resource, required: int) -> None:
        if gap < required:
            enforced = resource[0] not in {"xrf", "vmem"}
            out.append(Violation(rule, resource, a.name, b.name, gap, required, enforced))

    holds_a: dict[Resource, list[Hold]] = {}
    for h in a.holds:
        holds_a.setdefault(h.resource, []).append(h)
    for hb in b.holds:
        for ha in holds_a.get(hb.resource, ()):
            if gap + hb.first <= ha.last and gap + hb.last >= ha.first:
                if ha.phase is not None and hb.phase is not None and ha.phase == gap + hb.phase:
                    continue  # in-phase VectorFSM streams: one read serves both
                out.append(Violation("exclusive", hb.resource, a.name, b.name, gap,
                                     ha.last - hb.first + 1))

    gates_a: dict[Resource, list[Gate]] = {}
    for g in a.gates:
        gates_a.setdefault(g.resource, []).append(g)
    for pb in b.probes:
        for ga in gates_a.get(pb.resource, ()):
            if ga.first <= gap + pb.at <= ga.last:
                out.append(Violation("sequencer gate", pb.resource, a.name, b.name, gap,
                                     ga.last - pb.at + 1))

    rows_a: dict[tuple[Resource, int], list[RowAccess]] = {}
    for r in a.rows:
        rows_a.setdefault((r.resource, r.row), []).append(r)
    for rb in b.rows:
        for ra in rows_a.get((rb.resource, rb.row), ()):
            if rb.resource[0] == "mrf_phys":
                if ra.write != rb.write and ra.at == gap + rb.at:
                    out.append(Violation("same-row read/write collision", rb.resource + (rb.row,),
                                         a.name, b.name, gap, gap + 1))
                continue
            if ra.write and not rb.write:
                add("read after write", rb.resource + (rb.row,), ra.at - rb.at + 1)
            elif not ra.write and rb.write:
                add("write after read", rb.resource + (rb.row,), ra.at - rb.at)

    if b.reads & a.writes or b.writes & a.writes:
        add("pending write", ("mreg", *sorted((b.reads | b.writes) & a.writes)), a.writes_free)
    if b.writes & a.reads and not b.writer_ignores_readers:
        add("pending read", ("mreg", *sorted(b.writes & a.reads)), a.reads_free)
    return out


def gap_violations(a: Op, b: Op, gap: int, *, enforced_only: bool = False) -> list[Violation]:
    """Rules B breaks when issued ``gap`` cycles after A on an otherwise idle machine."""
    pa = profile(a)
    found = pair_violations(pa, profile(_with_port(b, [pa], [0], gap)), gap)
    return [v for v in found if v.enforced] if enforced_only else found


def min_gap(a: Op, b: Op, limit: int = 256, *, enforced_only: bool = False) -> tuple[int, list[Violation]]:
    """Smallest issue gap at which B may follow A from an idle machine, and the
    rules that bind at gap - 1 (empty when B may issue the very next cycle).
    Legality is not always monotone in the gap; ``gap_violations`` gives any gap."""
    for gap in range(1, limit + 1):
        if not gap_violations(a, b, gap, enforced_only=enforced_only):
            binding = gap_violations(a, b, gap - 1, enforced_only=enforced_only) if gap > 1 else []
            return gap, binding
    raise ValueError(f"{b.label} can never follow {a.label} within {limit} cycles")


def _with_port(op: Op, placed: Sequence[Profile], cycles: Sequence[int], at: int) -> Op:
    """Single pushes take ReadP1 when it is free at issue, otherwise ReadP0 (sequencer mux)."""
    if op.kind not in {"weight", "push_fp8"} or op.push_port is not None:
        return op
    p1 = ("mxu_port", op.mxu, "P1")
    busy = any(h.resource == p1 and t + h.first <= at <= t + h.last
               for p, t in zip(placed, cycles) for h in p.holds)
    return replace(op, push_port=0 if busy else 1)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

class Scheduler:
    """Places instructions in program order at their earliest legal issue cycle."""

    def __init__(self) -> None:
        self.placed: list[tuple[Op, Profile, int]] = []

    def _profiles(self) -> tuple[list[Profile], list[int]]:
        return [p for _, p, _ in self.placed], [t for _, _, t in self.placed]

    def violations(self, op: Op, at: int) -> list[Violation]:
        profiles, cycles = self._profiles()
        op = _with_port(op, profiles, cycles, at)
        pb = profile(op)
        out: list[Violation] = []
        for _, pa, ta in self.placed:
            out += pair_violations(pa, pb, at - ta)
        for cb in pb.counted:
            holders = [pa.name for _, pa, ta in self.placed for ca in pa.counted
                       if ca.resource == cb.resource and ta + ca.first <= at <= ta + ca.last]
            if len(holders) >= cb.capacity:
                out.append(Violation(f"{cb.capacity} in flight ({', '.join(holders)})",
                                     cb.resource, holders[-1], pb.name, at, at + 1))
        return out

    def _commit(self, op: Op, at: int) -> None:
        profiles, cycles = self._profiles()
        op = _with_port(op, profiles, cycles, at)
        if op.kind == "dma_wait":
            # The wait stalls until the channel's transfer has retired; later cycles are
            # relative to that release, and the channel is free from here on.
            for i, (o, p, t) in enumerate(self.placed):
                if o.kind == "dma" and o.channel == op.channel:
                    holds = [replace(h, last=at - t - 1) if h.resource == ("dma_channel", op.channel) else h
                             for h in p.holds]
                    self.placed[i] = (o, replace(p, holds=holds), t)
        self.placed.append((op, profile(op), at))

    def place(self, op: Op, not_before: int = 0, limit: int = 1024) -> tuple[int, list[Violation]]:
        """Issue ``op`` at the earliest cycle >= not_before (and after the previous
        instruction); returns the cycle and the rules that bound at cycle - 1."""
        start = max(not_before, (self.placed[-1][2] + 1) if self.placed else 0)
        last: list[Violation] = []
        for at in range(start, start + limit):
            found = self.violations(op, at)
            if not found:
                self._commit(op, at)
                return at, last
            if any(v.required >= FOREVER for v in found):
                raise ValueError(f"{op.label}: {found[0]}")
            last = found
        raise ValueError(f"{op.label} cannot issue within {limit} cycles of {start}")

    def check(self, op: Op, at: int) -> list[Violation]:
        """Place ``op`` at exactly ``at`` and return the rules it breaks (possibly none)."""
        found = self.violations(op, at)
        self._commit(op, at)
        return found


def schedule(ops: Iterable[Op]) -> list[tuple[Op, int, list[Violation]]]:
    """Earliest-issue schedule for ``ops`` in program order: (op, cycle, binding rules)."""
    sched = Scheduler()
    return [(op, *sched.place(op)) for op in ops]


def explain_schedule(ops: Iterable[Op]) -> str:
    """Human-readable schedule: cycle, DELAY to insert, and the binding rule for each op."""
    lines = []
    previous = None
    for op, at, binding in schedule(ops):
        gap = at - previous if previous is not None else None
        delay = "" if gap in (None, 1) else f"DELAY {gap - 2}"
        why = "; ".join(f"{v.rule} on {_fmt(v.resource)} vs {v.earlier}" for v in binding[:3])
        lines.append(f"T+{at:<4} {delay:<9} {op.label:<18} {why}")
        previous = at
    return "\n".join(lines)
