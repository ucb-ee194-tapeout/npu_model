# Registers and Execution State

## Scalar State

The scalar architectural state includes:

- `32` general-purpose integer registers, `x0` through `x31`
- one `32`-bit program counter `pc`

Rules:

- each scalar register stores one `32`-bit value
- `x0` is hardwired to zero
- `pc` stores the architectural instruction-word index
- under normal sequential execution, `pc` increments by `1`

## Tensor and Scale State

The tensor and scale architectural state includes:

- `64` tensor registers, `m0` through `m63`
- `32` scale registers, `e0` through `e31`
- two weight slots per MXU: `mxu0.w0`, `mxu0.w1`, `mxu1.w0`, `mxu1.w1`
- two accumulation buffers per MXU:
  `mxu0.acc0`, `mxu0.acc1`, `mxu1.acc0`, and `mxu1.acc1`

Tensor-register rules:

- each tensor register stores `32` rows
- each row stores `32` bytes
- each tensor register therefore stores `1024` raw bytes
- storage is flat and type-agnostic
- element interpretation is selected by instruction semantics

Whole-register views:

- one `m` register stores one `32 x 32 FP8_E4M3` tile
- one `m` register stores one `32 x 16 BF16` half-tile
- one full `32 x 32 BF16` tile occupies two consecutive `m` registers

Scale-register rules:

- each `e` register stores one `8`-bit scale payload
- one `e` register applies to one whole tensor operand
- `e` registers are distinct from both scalar `x` registers and tensor `m` registers

Accumulator-buffer rules:

- each accumulation buffer stores one `32 x 32 BF16` tile
- each accumulation buffer therefore stores `2048` raw bytes
- accumulation buffers are written by MXU instructions
- accumulation buffers are loaded from and stored to tensor registers only
- accumulation buffers are not directly addressed as `m` registers

## DMA and Control State

The DMA architectural state includes:

- one shared base address register, `dma.base`
- one busy / idle state bit per DMA channel

Rules:

- `dma.base` stores a `32`-bit address contribution used by `dma.load.chN` and
  `dma.store.chN`
- `dma.base` is shared across all channels
- `dma.wait.chN` observes the busy state of the selected channel only

The architecture-visible control plane includes at least:

- execution enable / halt control
- execution status / stop reason
- `pc` visibility
- DMA busy-state visibility for the eight architected DMA channels

## Memory Regions

The machine exposes three disjoint architectural memory regions.

| Region | Base Address | Size | Role |
|---|---:|---:|---|
| `IMEM` | `0x0002_0000` | `64 KiB` | Instruction memory |
| `VMEM` | `0x2000_0000` | `1 MiB` | On-chip tensor / vector data memory |
| `DRAM` | `0x8000_0000` | `16 GiB` | Off-chip backing data memory |

Rules:

- `IMEM` is word-addressed for architectural `pc` sequencing
- `VMEM` and `DRAM` are byte-addressed
- instruction fetch reads `IMEM`
- scalar data load / store access `VMEM` only
- `seld` reads one byte from `VMEM`
- `vload`, `vstore`, and `vload.weight.*` access `VMEM` only
- DMA is the only architected path between `DRAM` and `VMEM`

## Alignment Rules

The following alignment rules are architectural:

- instruction fetch address alignment: `4` bytes
- scalar `lb` / `lbu` / `sb` alignment: `1` byte
- scalar `lh` / `lhu` / `sh` alignment: `2` bytes
- scalar `lw` / `sw` alignment: `4` bytes
- `seld` alignment: `1` byte
- DMA source address alignment: `32` bytes
- DMA destination address alignment: `32` bytes
- DMA size granularity: multiple of `32` bytes
- `vload` and `vstore` VMEM address alignment: `32` bytes

## Initialization Rules

The architecture does not define deterministic reset contents for general data storage.

Unless software or host setup explicitly initializes them:

- `DRAM` contents are architecturally undefined
- `VMEM` contents are architecturally undefined
- scalar registers other than `x0` are architecturally undefined
- `e` registers are architecturally undefined
- tensor registers, MXU weight slots, MXU accumulation buffers, and `dma.base` are
  architecturally undefined

The host shall populate `IMEM` before enabling execution.

## Halt Model

Architecturally meaningful stop classes are:

- normal end-of-program completion
- `ecall`
- `ebreak`
- illegal instruction
- instruction-address misaligned
- misaligned scalar memory access
- illegal DMA issue
- other architecturally defined fatal model errors
