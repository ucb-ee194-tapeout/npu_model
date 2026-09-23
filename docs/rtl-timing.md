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

## DMA: unchanged approximation

DMA retains its existing model. Up to eight transfers queue in order, with only
the head progressing. For N bytes, the default transfer estimate is:

- Off-chip: `ceil((N + 8) / 4) * 2` core cycles.
- VMEM: `ceil(N / 64)` core cycles.
- Transfer: the maximum of those estimates, with a minimum of one cycle.

Thus a 1024-byte transfer takes 516 modeled execution cycles. Data moves as a
whole at completion, and the channel flag clears on the following tick.
The model does not simulate TileLink readiness, variable response latency or
beat-by-beat arbitration. DMA operands are read by its existing functional
implementation at completion; software must account for that limitation.
