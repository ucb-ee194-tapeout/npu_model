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

The baseline implementation supports concurrent long-chime unit activity.

Requirements:

- `mxu0`, `mxu1`, `vpu`, `xlu`, and DMA transfers may be active concurrently
- only one new instruction may issue in a cycle
- resource conflicts assert; only DELAY and DMA.WAIT stall the frontend
- issue does not perform dynamic reordering to bypass stalled older instructions
- execution timing is determined by frontend ordering, unit availability, architecturally
  defined blocking instructions, and fixed instruction latency classes

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
- matmul operations use the `95`-cycle latency class
- weight push, accumulator push and accumulator pop operations use the
  `33`-cycle latency class

`mxu1` requirements:

- internal `32 x 32` reduction-tree or equivalent throughput-matched fabric
- matmul operations use the `35`-cycle latency class
- weight push, accumulator push and accumulator pop operations use the
  `33`-cycle latency class

Scheduling constraints verified by the bank-conflict tests:

- when an MXU0 weight push from `m0` is followed by a matmul that reads `m0`,
  insert at least `delay 30` to avoid overlapping reads from the same physical
  MRF bank; `delay 29` still conflicts
- when an MXU0 matmul is followed by a pop from its accumulator, insert at
  least `delay 62` so accumulator row zero is available; `delay 61` is too short
- when `vadd.bf16` reading `m0` is followed by an MXU0 matmul reading `m0`,
  insert at least `delay 30`; `delay 29` still conflicts
- when `vtrpose.xlu` reading `m0` is followed by an MXU0 matmul reading `m0`,
  insert at least `delay 31`; `delay 30` still conflicts

These values are the immediate values of the `delay` instruction, which stalls
the following instruction by that many cycles. The minimums are covered by
`tests/test_bank_conflict.py`.

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
writes a BF16 pair. Pair fields name the low register; encoding register 63 as
the low half of a pair is illegal.

Timing requirements:

- pipelineable BF16 arithmetic, moves and transcendentals use the `66`-cycle
  latency class
- `vpack.bf16.fp8` uses the `66`-cycle latency class
- FP8-to-BF16 `vunpack.fp8.bf16` uses the `67`-cycle latency class
- non-pipelineable BF16 column-reduction operations (`vredsum.bf16`,
  `vredmin.bf16`, `vredmax.bf16`) use the `130`-cycle latency class
- BF16 row-reduction operations use the row latency classes:
  `vredsum.row.bf16` is `39` cycles, `vredmin.row.bf16` and
  `vredmax.row.bf16` are `34` cycles
- BF16 vector load-immediate operations (`vli.all`, `vli.row`,
  `vli.col`, `vli.one`) use operation-specific latency classes: `65` cycles
  for `vli.all` and `vli.row`, and `33` cycles for `vli.col` and `vli.one`
- the baseline lane count is `16 BF16` lanes
- operations that stream a full tile through the `16`-lane BF16 datapath use
  two internal half-tile passes over the architectural `32 x 32 BF16` tile

## Tensor Transform Unit

The XLU baseline implements:

- `vtrpose.xlu`

Timing requirement:

- each XLU operation uses the `66`-cycle latency class, reading source rows
  from T+1 through T+32 and writing destination rows from T+34 through T+65

## Structural-Conflict Handling

Resource conflicts are software-scheduled. In the current model, an illegal
overlap raises a conflict or scheduling error; units do not dynamically stall
or arbitrate to resolve it. `DELAY` provides explicit frontend spacing. While
an operation is in flight, each unit commits rows on its scheduled write edges
and holds its register reservations through the required access window.
