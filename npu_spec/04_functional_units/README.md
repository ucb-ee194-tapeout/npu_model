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

The frontend phase model used by the software model and trace infrastructure is:

- `IFU`: fetched-instruction stage
- `IDU`: decode and issue stage
- `EXU`: execution-unit stage

## Execution-Unit Overlap Model

The baseline implementation supports concurrent long-chime unit activity.

Requirements:

- `mxu0`, `mxu1`, `vpu`, `xlu`, and DMA transfers may be active concurrently
- only one new instruction may issue in a cycle
- issue stalls when the targeted unit cannot accept a new instruction
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

The microarchitecture preserves the architectural two-delay-slot rule without
speculation.

Implementation direction:

- the control block tracks unresolved control-flow shadows in program order
- sequential fetch continues until the youngest resolved shadow has observed its two
  required delay slots
- once the youngest resolved shadow has consumed its delay slots, its redirect is
  applied
- a branch or jump decoded in a delay-slot position is illegal and shall terminate
  execution before any younger redirect is applied

## Scalar Arithmetic and Logical Unit

The scalar path is responsible for:

- integer ALU operations
- branch and jump target generation
- scalar loads and stores to `VMEM`
- `seld` and `seli`
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
- resident local weight-slot source
- `BF16` architectural accumulation
- one local `32 x 32 BF16` accumulation buffer per MXU
- tensor-register-only accumulator preload and spill path
- local quantization path for `vmatpop.fp8.acc.*`
- ability to overlap with scalar and other long-chime units

`mxu0` requirements:

- internal `32 x 32` systolic fabric
- architectural `32 x 32` matmul implemented directly by that fabric
- deterministic launch latency class of `32` cycles

`mxu1` requirements:

- internal `32 x 32` reduction-tree or equivalent throughput-matched fabric
- deterministic launch latency class of `32` cycles

## Vector Processing Unit

The VPU baseline implements the following operations:

- `vadd.bf16`
- `vredsum.bf16`
- `vsub.bf16`
- `vminimum.bf16`
- `vmaximum.bf16`
- `vmul.bf16`
- `vmov`
- `vrecip.bf16`
- `vexp.bf16`
- `vrelu.bf16`
- `vsquare.bf16`
- `vcube.bf16`

Architectural BF16 operand model:

- each BF16 VPU instruction consumes a full `32 x 32 BF16` tile from the named
  source register pair `{m[vs], m[vs + 1]}`
- each BF16 VPU destination names the low register of the destination pair
  `{m[vd], m[vd + 1]}`
- this makes BF16 VPU register-pair usage match the existing
  `vmatpop.bf16.acc.*` / `vmatpush.acc.bf16.*` accumulator transfer convention

Timing requirements:

- pipelineable BF16 operations use the `4`-cycle latency class
- non-pipelineable BF16 operations such as `vexp` and `vrecip.bf16` use the
  `16`-cycle latency class
- the baseline lane count is `16 BF16` lanes
- the `16`-lane datapath completes one BF16 VPU instruction as two internal
  half-tile passes over the architectural `32 x 32 BF16` tile

## Tensor Transform / Reduction Unit

The XLU baseline implements:

- `vtrpose.xlu`
- `vreduce.max.xlu`
- `vreduce.sum.xlu`

Timing requirement:

- each XLU operation uses the `4`-cycle latency class

## Structural-Conflict Handling

Structural conflicts shall be handled by stalls or arbitration.

They shall not create:

- partial architectural row retirement
- partial architectural tile retirement
- architecturally visible younger-over-older preemption
