# RTL timing model and validation

The model follows the default Atlas geometry and scalar/tensor pipelines in
`../src/main/scala`. Regression tests compare actual Verilator traces with model
cycles and output bits. All registered workloads are exercised as well.
[Fixture generation and coverage](../tests/rtl/README.md) documents the harnesses
and their boundaries. These are directed equivalence checks, not a proof of all
possible full-system executions.

## Scalar pipeline and PCs

The reference is `atlas/scalar/ScalarCore.scala` and `PcControl.scala`. A tick
models one clock edge. Cycle 1 follows host start and fetches word 0, which
executes in cycle 2. Decode, scalar execution/writeback and engine launch share
stage S1.

Taken branches and jumps have **one delay slot**. A not-taken branch does not
mark its successor as a delay slot. PCs and jump links count words. Assembly
branch and JAL offsets count words; direct Python instruction constructors hold
the encoded immediate, which the RTL shifts right by one. JALR adds its signed
12-bit immediate to its source register as a word address.

`delay N` issues first, then stalls the following S1 instruction for N cycles.
DMA.WAIT holds S1 while its channel is busy. Halt detection precedes stalls.
`Core.last_cycle` exposes PC, issue, stall, redirect and halt observations.

## Instruction memory

`InstrMem.scala` has **one 128 KiB synchronous 1R1W memory**, without double
buffering. The model has one live instruction bank and holds the S1 instruction
during frontend stalls. Host reads share the fetch read port; host writes use
the independent write port. Undefined same-address read/write collisions raise
an error.

Fetch indexes the live memory using the low 15 PC bits. Host-written instructions
beyond the initially loaded program are executable, and high architectural PCs
alias the physical memory without truncating the architectural PC or jump links.
The host interface accepts decoded instruction objects. Uninitialized words end
a model program as a software convenience; real software should use ECALL/EBREAK.

## LSU and tensor engines

For a command issued at T (default 32 rows):

| Operation | First write | Last write |
| --- | --- | --- |
| Scalar store | T+1 | T+1 |
| Scalar load register writeback | T+3 | T+3 |
| VLOAD / VSTORE | T+3 | T+34 |
| MXU weight/accumulator push, accumulator pop | T+1 | T+32 |
| MXU0 systolic compute, accumulator rows | T+63 | T+94 |
| MXU1 inner-product compute, accumulator rows | T+3 | T+34 |
| VPU ordinary BF16 operations | T+2 | T+65 |
| VPU row min/max | T+2 | T+33 |
| VPU row sum | T+7 | T+38 |
| VPU column reductions | T+66 | T+129 |
| VPU pack | T+3 | T+65, every other cycle |
| VPU unpack | T+3 | T+66 |
| VLI all/row | T+1 | T+64 |
| VLI column/one | T+1 | T+32 |
| XLU transpose | T+34 | T+65 |

The MXU table lists row-write edges; command issue checks can impose additional
spacing when operations reuse a port, register or accumulator. The bank-conflict
tests distinguish these cases: MXU0 can read a weight slot while the push is
still streaming when the MRF sources differ, while MXU1 requires `delay 30`
before reading the slot. A weight push and matmul that both read `m0` need
`delay 30` because their physical MRF reads overlap; `vadd.bf16` reading `m0`
followed by that matmul also needs `delay 30`, and XLU transpose reading `m0`
needs `delay 31`. After a matmul writes `acc0`, a pop or another accumulate
matmul that reads `acc0` needs `delay 62`. Each accumulator has one read port,
so a pop and an accumulate matmul cannot stream from the same accumulator at
once; an overwrite matmul does not read it and may issue the cycle after a pop
of the same accumulator (the `perf_mm_*` baremetal schedule). A VPU or XLU write to `m4` followed
by a matmul reading `m4` needs `delay 64`. `tests/test_bank_conflict.py` checks
each minimum and the one-cycle-short case. The `DELAY` immediate is not the
issue-cycle gap: the `DELAY` instruction itself issues before its stall begins.


Scalar loads have no same-cycle bypass. Scalar, vector-load and vector-store
paths progress independently. Sources are sampled at their row-read edges;
results become visible row by row. Software scheduling violations raise errors.
VPU has two independent single-input slots; binary and row-reduction operations
occupy both. `issue_busy_mask` exposes the opcode-specific issue restrictions.

Scalar LSU addresses are **bytes**. VLOAD/VSTORE bases are **words**, and their
immediate contributes 32 words per unit. For example, byte address 0x2000 needs
base 0x800; an additional 1024 bytes uses immediate 8. VMEM contains six contiguous
256 KiB banks. Logical MREG reservations distinguish readers and writers, while
physical port checks account for mN/m(N+32) sharing a 1R1W bank.

## Numerical behavior and layouts

- Eleven unary BF16 operations use exhaustive RTL-generated lookup tables:
  reciprocal, sqrt, sin, cos, tanh, log2, exp, exp2, square, cube and ReLU. Every
  possible BF16 input encoding is represented, including special values.
- Add/subtract and column sums truncate the FP32 result to BF16, with HardFloat's
  canonical NaN. Min/max use the RTL's ordered-bit comparisons.
- MXU0 uses the default custom FP8×FP8+BF16 FMA, rounding each MAC. MXU1 uses the
  default 32-bit anchor accumulator with seven bits of exponent headroom. These
  paths use integer arithmetic and are compared to actual RTL outputs.
- Weight-buffer rows are output **columns**, so row-major B tiles must be
  transposed before weight push for A×B.
- VPU pack concatenates successive 16-lane rows from its bank stream. MXU BF16
  push/pop instead joins matching rows of the even/odd banks into 32 columns.
  Attention kernels use an MXU accumulator round trip to quantize that layout.
- E8M0 unit scale is **127**. VPU pack divides by the scale; MXU pop multiplies.
  Their RTL converters also differ at the rounded FP8 0x7f encoding. The model
  preserves both behaviors, including the MXU converter emitting that encoding.

The supplied assembly now uses correct VLS address units, even BF16 register
pairs, compatible layouts and legal physical-bank schedules. All `.bin`/`.hex`
images are regenerated. Affected workload references explicitly use RTL rounding
and approximation rules; ideal PyTorch mathematical results can differ.

## Validation boundaries

Scalar, VPU, both MXUs, LSU and XLU have real-RTL comparison fixtures. Engine
harnesses supply synchronous memory responses and check cycle/address/data
traces; VPU traces also check busy and opcode-specific issue-busy signals.
IMEM host arbitration and CSR counters have directed Python tests. The harnesses
do not instantiate a complete AtlasTile/TileLink system. The default geometry,
custom-FMA systolic architecture and two-stage IPT are the supported timing
configuration.

## DMA: beat-level engine, parameterized memory

[`dma.py`](../npu_model/hardware/dma.py) follows `diplomatic/memory/DMA.scala`
beat for beat. For a transfer launched at T:

- The command occupies one of eight slots from T+1. Its channel reads busy to
  S1 from T+1 until the cycle after the slot retires.
- Loads issue one 32-byte TileLink Get per cycle from T+1, for the oldest
  undispatched slot, while fewer than 64 beats are outstanding, the next
  source ID is free, and the memory accepts (`tl.a.ready`).
- Stores first read VMEM one line per cycle from T+1, when the bank grants it.
  A line read at cycle C can be requested as a Put at C+2 (SyncReadMem, then
  one cycle in the store data queue), so the first Put of a store issues at
  T+3 at the earliest.
- Responses may return in any order. Load data is written to VMEM in the
  response cycle if the bank grants the write; otherwise channel D is held,
  and every response behind it waits. Store acknowledgements retire freely.
- A slot retires in the first cycle in which it has issued every beat and
  has none outstanding. Slots can retire out of enqueue order.

Beats of the next command issue while the previous command's responses are
still returning, so back-to-back commands are throughput-bound.

VMEM grants follow `Vmem.scala`: an LSU scalar or vector access to a bank in a
cycle denies the DMA that bank, and a DMA write beats a DMA read to the same
bank. The LSU publishes its next-cycle bank accesses through the conflict
checker so the grant does not depend on the Python unit order.

Register operands are the values S1 read at launch; `dma.base` is the value
registered at launch (`dma.config` writes it at issue, without occupying the
engine or a channel). Source bytes are sampled when each beat is read (DRAM at
the Get, VMEM at the granted line read), and each beat's destination bytes are
written as the RTL would write them. Sizes must be non-zero multiples of 32
bytes up to 4 KiB, as the RTL asserts.

What sits behind the TileLink port is not fixed RTL and lives in
[`memory_backend.py`](../npu_model/hardware/memory_backend.py). The supported
target is VCS simulation of EE290SimConfig, where the port reaches DRAMSim2
(Chipyard's `make run-binary` always passes `+dramsim`; the device is
`DDR3_micron_64M_8B_x4_sg15`, open-page, clocked at 666 MHz against the
500 MHz core) through TileLink buffers and a 64-bit AXI memory port.

The default backend, `curve`, is the Mess simulator's bandwidth-latency
model (Esmaili-Dokht et al., MICRO 2024) ported to this interface: the memory
is described only by measured curves of latency against delivered bandwidth,
one per read percentage, in Mess's JSON curve format. A controller measures
the bandwidth delivered over a window of beats, selects the curve by the mix
of the beats being issued, looks up the latency and paces responses at the
curve's peak bandwidth. Because the DMA engine keeps 64 beats in flight
regardless of latency, a probe-measured curve rises steeply at the peak only
because the probe waited behind the engine's own backlog, which the pacing
already reproduces; `latency_mode="knee"` reads the curve no further than
90 % of the peak, `"unloaded"` applies only each curve's lead-off latency,
and `"mess"` applies it verbatim with Mess's overflow penalty. Curve files
live in `npu_model/configs/memory_curves`. The default,
`ee290sim_vcs_probe.json`, was measured on VCS with DRAMSim2 refresh
disabled (`../baremetal/dramsim2_ini_norefresh`):
`scripts/gen_dma_probe_programs.py` writes the DMA-side equivalent of the
Mess benchmark (one timed 32 B probe against background traffic from the
other channels at a sweep of mixes and loads) and
`scripts/fit_dma_probe_curve.py` turns its logs into a curve file, tagged
`latencyMode: "unloaded"` because a DMA probe's loaded latency is only its
wait behind the engine's backlog. The file gives a lone-beat round trip of
46 cycles per Get and 42 per Put and a streaming rate of about 23 cycles per
32-byte beat, loads and stores sharing it; those numbers include real DRAM
timing (DDR3 row activate, CAS latency, low-power exit and the 666/500 MHz
clock crossing), not just the fabric. Unfitted, it lands within 85 cycles of
every program in `../baremetal/assembly/perf_dma_*.S` (which bank the
mcycles delta around a DMA region into CSR_DBG1), a total of 656 cycles over
18 programs; `scripts/calibrate_dma_backend.py` reports the residual. Refresh
is excluded because it adds 0 to about 130 cycles per transfer window
depending on the absolute simulation time at which the program starts; runs
with the default DRAMSim2 ini carry that offset on top (see the calibration
notes in the README). The same probe programs measure the bringup board later
without code changes.

A second backend, `fixed`, is a rate-limited link with a constant or scripted
per-beat latency. It is not a target: the RTL trace test uses it to script
out-of-order responses and `tl.a.ready` denials against the engine, and the
calibration script can sweep it.
`HardwareConfig.dma_memory_backend` and `dma_memory_params` select and
parameterize the backend.

Operands follow AtlasCore and ScalarCore exactly:

- The VMEM operand is a 32-bit **word** address. The engine receives
  `vmemAddr(wordAddrBits-1, wordOffBits)` as the line, so the low three bits
  are ignored and bits above 19 (for 1.5 MiB) wrap; the engine asserts when
  the line range leaves VMEM, which the model raises. The supplied workloads
  were migrated from byte to word operands (and their derived `vload`/`vstore`
  bases from `srli ..., 2` to a move).
- The off-chip address is the 64-bit `{dma.base, x[rs]}`. DRAM is mapped as
  EE290SimConfig maps it, 64 GiB at `0x8000_0000` (`ArchStateConfig.dram_base`,
  `dram_size`), held in a page-sparse store so the window costs nothing until
  touched. A transfer outside the window raises; on the RTL it would leave the
  tile for whatever the fabric maps there. Baremetal programs, which address
  DRAM at `0x9000_0000` with base 0, run unmodified. The supplied workloads
  program base 1 and keep their layouts at `Program.dram_base = 4 GiB`.
- The size is `x[rs2]` in bytes; the RTL truncates it to 13 bits and asserts
  on zero beats or a range past VMEM. The model raises for anything that is
  not a non-zero multiple of 32 up to 4 KiB.
