# Functional Units

## Top-Level Organization

The baseline microarchitecture is organized around:

- an instruction frontend
- a scalar execution path
- a scale register file
- a tensor register file and tensor interconnect
- two MXUs
- one VPU
- one XLU
- an instruction-memory path
- a VMEM subsystem
- a DMA engine complex connecting `DRAM` and `VMEM`
- a host-visible control and status block

## Frontend

The frontend baseline is:

- single-stream instruction fetch
- single instruction decode
- single issue decision per cycle
- fixed-width `32`-bit fetch from `IMEM`

The RTL has two pipeline stages: synchronous IMEM fetch, followed by combined
decode, register read, scalar execute/writeback, and engine launch. Trace labels
for decode and engine execution do not imply an extra pipeline register.

## Execution-Unit Overlap Model

The baseline implementation supports concurrent long-running unit activity.

Requirements:

- `mxu0`, `mxu1`, `vpu`, `xlu`, and DMA transfers may be active concurrently
- only one new instruction may issue in a cycle
- resource conflicts assert; only DELAY and DMA.WAIT stall the frontend
- issue does not perform dynamic reordering to bypass stalled older instructions
- execution timing is determined by frontend ordering, the two blocking
  instructions (`DELAY`, `DMA.WAIT`), and fixed per-operation timing

## Timing Reference

### Latency classes

A **latency class** is the number of cycles from issue through the last
result write, counting the issue cycle: an operation issued at T writes its
last result at T + latency - 1. It is not the required gap before a dependent
instruction; see Spacing instructions with `DELAY`.

| Functional unit | Operation | Latency class |
| --- | --- | ---: |
| Scalar | ALU, branch, CSR, and `delay` instructions | 1 cycle |
| LSU | Scalar loads (`lb`, `lh`, `lw`, `lbu`, `lhu`, `seld`) | 4 cycles |
| LSU | Scalar stores (`sb`, `sh`, `sw`) | 2 cycles |
| LSU | `vload`, `vstore` | 35 cycles |
| MXU0 | Matmul and matmul-accumulate | 95 cycles |
| MXU0 | Weight push, accumulator push, accumulator pop | 33 cycles |
| MXU1 | Matmul and matmul-accumulate | 35 cycles |
| MXU1 | Weight push, accumulator push, accumulator pop | 33 cycles |
| VPU | Binary/unary BF16 arithmetic, moves, and transcendentals | 66 cycles |
| VPU | `vredsum.bf16`, `vredmin.bf16`, `vredmax.bf16` | 130 cycles |
| VPU | Row sum (`vredsum.row.bf16`) | 39 cycles |
| VPU | `vredmin.row.bf16`, `vredmax.row.bf16` | 34 cycles |
| VPU | `vpack.bf16.fp8` | 66 cycles |
| VPU | `vunpack.fp8.bf16` | 67 cycles |
| VPU | `vli.all`, `vli.row` | 65 cycles |
| VPU | `vli.col`, `vli.one` | 33 cycles |
| XLU | `vtrpose.xlu` | 66 cycles |
| DMA | `dma.config` | 1 cycle |
| DMA | `dma.load`, `dma.store` | Variable; see formula below |
| Frontend | `dma.wait.chN` | Waits until that channel completes |

For the default hardware configuration, a DMA transfer of `N` bytes takes
`max(2 * ceil((N + 8) / 4), ceil(N / 64))` cycles, with a minimum of one cycle.

### Spacing instructions with `DELAY`

Units never stall each other. An instruction that issues too early raises a
conflict or scheduling error (see Structural-Conflict Handling), so software
spaces dependent instructions with `DELAY`. Three facts determine the spacing.

**1. `DELAY` arithmetic.** If instruction A issues at cycle T, the next
instruction B issues at:

- T+1 when B directly follows A;
- T+N+2 when `DELAY N` sits between them (`DELAY` issues at T+1, then holds B
  in S1 for N cycles).

So for a required issue gap G (B's issue cycle minus A's), the minimum is
`DELAY G-2`, and G = 1 needs no `DELAY`.

**2. The wait is not the latency.** A latency class gives the cycle of A's last
write, T + latency - 1. B only waits for the resource it actually shares with A,
and only until the cycle at which B first uses it. Every unit reads and writes
row by row, and many touch their first row a cycle or more after issue, so G is
often shorter than A's latency. Counting cycles from A's issue:

```
G = (cycle A last touches it) - (offset after its own issue at which B first touches it) + 1
```

"It" is a register bank for a port conflict, and a row for data that A produces
and B consumes. When both stream one row per cycle, row 0 decides. A
reservation instead requires B to issue no earlier than its free-from cycle.

**3. What counts as shared.** Four kinds of check apply:

- **Register reservations**, checked when B issues: B may not read or write a
  register that A will write, or write a register that A still reads (VLOAD is
  exempt from the latter). Each reservation ends at a fixed cycle (last column
  of the access-window table); B may issue from that cycle on.
- **Physical register ports**, checked every cycle: each physical bank has one
  read port and one write port, and a read and write of the same row in the
  same cycle is an error. `mN` and `m(N+32)` share a physical bank.
- **Same-unit issue rules**: a unit busy with A may refuse B, even when they
  use no common register.
- **MXU buffer rules** for weight slots and accumulators.

### Access windows

Offsets from issue cycle T, default 32-row geometry. "Free from" is the first
issue cycle at which another instruction may take a conflicting reservation on
the register.

| Operation | Reads | Writes | Free from |
| --- | --- | --- | --- |
| VPU binary/unary BF16, `vmov` | `vs`: T..T+31, `vs+1`: T+32..T+63 (same for a `vs2` pair) | `vd`: T+2..T+33, `vd+1`: T+34..T+65 | reads T+64, writes T+66 |
| VPU `vredmin.row`, `vredmax.row` | `vs`, `vs+1`: T..T+31 | `vd`, `vd+1`: T+2..T+33 | reads T+32, writes T+34 |
| VPU `vredsum.row` | `vs`, `vs+1`: T..T+31 | `vd`, `vd+1`: T+7..T+38 | reads T+32, writes T+39 |
| VPU `vredsum`, `vredmin`, `vredmax` | `vs`: T..T+31 and T+64..T+95, `vs+1`: T+32..T+63 and T+96..T+127 | `vd`: T+66..T+97, `vd+1`: T+98..T+129 | reads T+128, writes T+130 |
| `vpack.bf16.fp8` | `vs2`: T..T+31, `vs2+1`: T+32..T+63 | `vd`: every other cycle, T+3..T+65 | reads T+64, writes T+66 |
| `vunpack.fp8.bf16` | `vs2`: T..T+31 | `vd`: T+3..T+34, `vd+1`: T+35..T+66 | reads T+32, writes T+67 |
| `vli.all`, `vli.row` | none | `vd`: T+1..T+32, `vd+1`: T+33..T+64 | T+65 |
| `vli.col`, `vli.one` | none | `vd`: T+1..T+32 | T+33 |
| `vtrpose.xlu` | `vs1`: T+1..T+32 | `vd`: T+34..T+65 | reads T+34, writes T+66 |
| MXU weight or accumulator push | `vs1` (and `vs1+1` for BF16): T..T+31 | weight or accumulator rows T+1..T+32 | T+33 |
| MXU accumulator pop | accumulator rows T..T+31 | `vd` (and `vd+1` for BF16): T+1..T+32 | T+33 |
| MXU0 matmul | `vs1`: T..T+31; accumulator rows T..T+31 for `.acc`; weight row c first at T+1+c | accumulator rows T+63..T+94 | T+33 |
| MXU1 matmul | `vs1`: T..T+31; accumulator rows T..T+31 for `.acc`; whole weight slot T+1..T+32 | accumulator rows T+3..T+34 | T+33 |
| `vload` | VMEM rows T+1..T+32 | `vd`: T+3..T+34 | T+35 |
| `vstore` | `vd`: T+1..T+32 | VMEM rows T+3..T+34 | T+35 |
| Scalar load | VMEM word T+1 | `rd` T+3 (no bypass) | not applicable |
| Scalar store | none | VMEM T+1 | not applicable |

### Same-unit issue spacing

| A, then B on the same unit | Minimum gap G | Minimum `DELAY` |
| --- | --- | --- |
| VPU op, then a binary or row-reduction op, or the reverse | latency of A - 1 | latency of A - 3 |
| VPU op, then the same op or one from its logic group* | latency of A - 1 | latency of A - 3 |
| VPU single-input op, then a single-input op from another group | 1 | none (two VPU slots) |
| XLU, then XLU | 66 | 64 |
| `vload`, then `vload`; `vstore`, then `vstore` | 35 | 33 |
| Scalar load, then scalar load | 3 | 1 |
| Matmul, then matmul (same MXU) | 32 | 30 |
| BF16 accumulator push, then matmul (same MXU) | 32 | 30 |
| Pop, then pop (same MXU) | 32 | 30 |
| Weight push, then weight push (same MXU) | 32 | 30 |
| Accumulator push, then accumulator push (same MXU, any accumulator) | 32 | 30 |

\* Logic groups: {`vadd`, `vsub`, `vredsum.row`}, {`vexp`, `vexp2`},
{`vsin`, `vcos`}, {`vsquare`, `vcube`}, {`vmaximum`, `vredmax`},
{`vminimum`, `vredmin`}, and all `vli` forms. A binary or row-reduction op
occupies both VPU slots; at most two VPU ops are in flight. `vload` and
`vstore` use independent paths. On each MXU, weight pushes share one write
path and accumulator pushes another, which carries one push at a time.

### MXU buffer rules

| A, then B (same MXU) | MXU0 gap | MXU1 gap |
| --- | --- | --- |
| Weight push to `wN`, then matmul reading `wN` | 1: rows arrive before the array reaches them | 32: the whole slot is read from B's T+1 |
| Matmul reading `wN`, then weight push to `wN` | 63 | 32 |
| Matmul writing `accN`, then pop or `matmul.acc` of `accN` | 64: B reads row r at its T+r, after row r lands | 32: a pop may not read `accN` while the matmul does, and a second matmul waits for the read port |
| FP8 accumulator push to `accN`, then matmul on `accN`‡ | 2: B reads row r one cycle after it lands | 33 |
| Accumulator push to `accN`, then pop of `accN` | 2 | 2 |

‡ A BF16 accumulator push holds both MXU read ports, so any matmul after it
waits 32 cycles on either MXU.

### Tested examples

`tests/test_bank_conflict.py` checks that each minimum below passes and, except
for the first row, that one cycle less raises (for `DELAY 0`, one cycle less
means no `DELAY`). Cycles count from A's issue T; B issues at T+G.

| A, then B | Shared resource and binding cycles | G | Minimum `DELAY` |
| --- | --- | ---: | ---: |
| MXU0 weight push `m0` to `w0`, then MXU0 matmul from `m2` reading `w0` | Weight row c lands at T+1+c; B first uses it at T+G+1+c | 1 | none |
| MXU1 weight push `m0` to `w0`, then MXU1 matmul from `m2` reading `w0` | Last weight row lands at T+32; B reads the whole slot from T+G+1 | 32 | 30 |
| MXU0 weight push from `m0`, then MXU0 matmul reading `m0` | Read port of `m0`: A's last read T+31; B's first read T+G | 32 | 30 |
| `vadd.bf16` reading `m0`, then MXU0 matmul reading `m0` | Read port of `m0`: A's last read T+31 | 32 | 30 |
| `vtrpose.xlu` reading `m0`, then MXU0 matmul reading `m0` | Read port of `m0`: A's last read T+32 | 33 | 31 |
| MXU0 matmul writing `acc0`, then pop of `acc0` | Row 0 lands at T+63; B reads row 0 at T+G | 64 | 62 |
| MXU0 matmul writing `acc0`, then `matmul.acc` of `acc0` | Row 0 lands at T+63; B reads row 0 at T+G | 64 | 62 |
| `vadd.bf16` writing `m4`, then MXU0 matmul reading `m4` | Reservation on `m4`, free from T+66 | 66 | 64 |
| `vtrpose.xlu` writing `m4`, then MXU0 matmul reading `m4` | Reservation on `m4`, free from T+66 | 66 | 64 |
| MXU0 weight push to `w0`, then MXU0 weight push to `w1` | Weight write path: A's last row at T+32; B's first at T+G+1 | 32 | 30 |
| MXU1 accumulator push to `acc0`, then to `acc1` | Accumulator write path: A's last row at T+32; B's first at T+G+1 | 32 | 30 |
| MXU0 FP8 accumulator push to `acc0`, then `matmul.acc` of `acc0` | Row r lands at T+1+r; B reads it at T+G+r | 2 | 0 |
| MXU1 FP8 accumulator push to `acc0`, then pop of `acc0` | Row r lands at T+1+r; B reads it at T+G+r | 2 | 0 |

## Decode Responsibilities

The decode path shall recognize:

- scalar `R`, `I`, `S`, `SB`, `U`, and `UJ` instructions
- tensor `VLS`, `VR`, and `VI` instructions
- scalar `delay`
- DMA transfer and DMA control families

Minimum decode outputs include:

- scalar fields: `rd`, `rs1`, `rs2`, immediate
- tensor fields: `vd`, `vs1`, `vs2`, subopcode
- reconstructed weight-slot and DMA-channel selectors where applicable
- target execution unit
- legality classification

Decode must enforce the reserved-zero rules defined by the ISA, including:

- unary `VPU` operations: `vs2 = 0`
- `XLU` operations: `vs2 = 0`
- `vmatpush.weight.*`: `vs2 = 0` and `vd[5:1] = 0`
- `vmatpush.acc.*`: `vs2 = 0` and `vd[5:1] = 0`
- `vmatpop.*`: `vs1 = 0` and `vs2[5:1] = 0`
- `vmatmul.*`: `vd[5:1] = 0` and `vs2[5:1] = 0`
- `dma.config.chN` and `dma.wait.chN`: `rd = x0`
- `dma.wait.chN`: `rs1 = x0`

Decode must also enforce pair-register legality for instructions whose semantics
consume or produce `{m[r], m[r + 1]}`:

- the encoded tensor register field names the low register of the pair
- encoding `r = 63` for such a field is illegal

## Delay-Slot Handling

The RTL has one delay slot for taken branches and jumps. The sequential word
already being fetched executes once before the redirect target. A branch or jump
in that slot halts as illegal; a not-taken branch does not mark a delay slot.
The slot marker and S1 PC hold across DELAY and DMA.WAIT stalls.

## Scalar Arithmetic and Logical Unit

The scalar path is responsible for:

- integer ALU operations
- branch and jump target generation
- scalar loads and stores to `VMEM`
- `seld` and `seli`
- scalar `delay`
- `dma.base` programming
- halt-status generation

The intended scalar implementation slice is partitioned into:

- scalar decoder
- scalar register file
- scale-register write path
- scalar ALU / compare datapath
- branch and jump target unit
- VMEM-facing scalar LSU
- scalar control block

## Matrix Execution Units

The two MXUs share the same architectural interface but intentionally differ internally.

Baseline intent:

- `mxu0`: systolic-array accumulation
- `mxu1`: inner-product-tree accumulation

Shared requirements:

- whole-register activation source
- two resident local weight slots per MXU
- `BF16` architectural accumulation
- two local `32 x 32 BF16` accumulation buffers per MXU
- tensor-register-only accumulator preload and spill path
- local quantization path for `vmatpop.fp8.acc.*`
- ability to overlap with scalar and other long-chime units

`mxu0` requirements:

- internal `32 x 32` systolic fabric
- architectural `32 x 32` matmul implemented directly by that fabric

`mxu1` requirements:

- internal `32 x 32` reduction-tree or equivalent throughput-matched fabric

## Vector Processing Unit

The VPU baseline implements:

- binary BF16 operations: `vadd.bf16`, `vsub.bf16`, `vmul.bf16`,
  `vminimum.bf16`, `vmaximum.bf16`
- unary BF16 operations: `vmov`, `vrecip.bf16`, `vsqrt.bf16`, `vsin.bf16`,
  `vcos.bf16`, `vtanh.bf16`, `vlog2.bf16`, `vexp.bf16`, `vexp2.bf16`,
  `vrelu.bf16`, `vsquare.bf16`, `vcube.bf16`
- BF16 reductions: `vredsum.bf16`, `vredmin.bf16`, `vredmax.bf16`,
  `vredsum.row.bf16`, `vredmin.row.bf16`, `vredmax.row.bf16`
- format conversion: `vpack.bf16.fp8`, `vunpack.fp8.bf16`
- immediate writes: `vli.all`, `vli.row`, `vli.col`, `vli.one`

Ordinary BF16 tile operations read their input from a register pair
`{m[vs], m[vs + 1]}` and write their result to a destination pair
`{m[vd], m[vd + 1]}`. There are operation-specific exceptions: VLI operations
have no source, `vli.col` and `vli.one` write a single register, `vpack` reads a
BF16 pair and writes one FP8 register, and `vunpack` reads one FP8 register and
writes a BF16 pair. Pair fields name the low register, which the VPU requires
to be even.

Latency classes are listed in the Timing Reference.

Datapath details:

- the baseline lane count is `16 BF16` lanes
- operations that stream a full tile through the `16`-lane BF16 datapath use
  two internal half-tile passes over the architectural `32 x 32 BF16` tile

## Tensor Transform Unit

The XLU baseline implements:

- `vtrpose.xlu`

Its latency class and access windows are listed in the Timing Reference.

## Structural-Conflict Handling

Resource conflicts are software-scheduled. In the current model, an illegal
overlap raises a conflict or scheduling error; units do not dynamically stall
or arbitrate to resolve it. `DELAY` provides explicit frontend spacing. While
an operation is in flight, each unit commits rows on its scheduled write edges
and holds its register reservations through the required access window.
