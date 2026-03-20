# Memory Model

## High-Level Rule

The baseline memory system keeps the asynchronous boundary narrow:

- `IMEM` fetch is local and deterministic
- `VMEM` is the sole on-chip tensor staging memory
- `DRAM` access is off-chip and asynchronous
- DMA is the only `DRAM <-> VMEM` path

## Blocking On-Chip Transfers

The following instructions are architecturally blocking:

- scalar `lb`, `lh`, `lw`, `lbu`, `lhu`, `sb`, `sh`, `sw`
- `seld`
- `vload`
- `vstore`
- `vmatpush.weight.mxu0`
- `vmatpush.weight.mxu1`
- `vmatpush.acc.fp8.mxu0`
- `vmatpush.acc.fp8.mxu1`
- `vmatpush.acc.bf16.mxu0`
- `vmatpush.acc.bf16.mxu1`
- `vmatpop.fp8.acc.mxu0`
- `vmatpop.fp8.acc.mxu1`
- `vmatpop.bf16.acc.mxu0`
- `vmatpop.bf16.acc.mxu1`

The intent is to keep on-chip movement deterministic and straightforward to verify.

## VMEM Ordering Rules

The cycle-accurate model shall preserve VMEM ordering across units:

- a VMEM reader shall not observe data older than the most recent completed VMEM writer
- `dma.store.chN` shall not complete before older blocking VMEM writes make their data
  architecturally visible
- this ordering is modeled with fixed completion timing and program order, not with a
  general architectural dependency scoreboard

## Asynchronous DMA

DMA is channelized and asynchronous.

Each channel supports:

- at most one outstanding transfer
- independent busy / completion state
- `dma.wait.chN` synchronization

The microarchitecture may implement the DMA channels with shared internal data paths or
arbitration, provided the architecture-visible channel behavior is preserved.

`dma.wait.chN` and scalar `delay` behave as frontend fences:

- neither instruction allocates a normal execute-stage slot while it is holding decode
- if channel `N` is already done when `dma.wait.chN` reaches decode, the instruction
  spends that cycle in decode and retires directly
- if channel `N` is not yet done, decode remains occupied until the transfer completes,
  then the instruction retires directly from decode
- younger instructions shall not issue past that decode fence until the wait retires
- `delay N` also remains resident in decode for exactly `N` additional core cycles after
  its decode cycle, then retires directly

## DMA Addressing and Regions

DMA transfer instructions form addresses as follows:

- off-chip address contribution comes from `x[...] + dma.base`
- on-chip address contribution comes from scalar register operands naming `VMEM`
  locations

DMA rules:

- DMA is the only architected `DRAM <-> VMEM` path
- a DMA issue to a busy channel is illegal
- DMA source and destination addresses must be `32`-byte aligned
- DMA sizes must be multiples of `32` bytes

## Baseline Transfer Formulas

Definitions:

- `OFFCHIP_BYTES_PER_BEAT = OFFCHIP_LINK_WIDTH_BITS / 8 = 4`
- `VMEM_BYTES_PER_BEAT = VMEM_BUS_WIDTH_BITS / 8 = 64`

Required formulas:

- `dma_offchip_cycles(bytes) = ceil((bytes + 4 * DMA_OFFCHIP_COMMAND_WORDS) / OFFCHIP_BYTES_PER_BEAT) * OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`
- `vmem_transfer_cycles(bytes) = ceil(bytes / VMEM_BYTES_PER_BEAT) * VMEM_BUS_CORE_CYCLES_PER_BEAT`
- `dma_transfer_cycles(bytes) = max(dma_offchip_cycles(bytes), vmem_transfer_cycles(bytes))`

For the frozen baseline values:

- one off-chip beat costs `2` core cycles
- one VMEM beat costs `1` core cycle
- one `vload` or `vstore` of a `1024`-byte tensor register takes `16` cycles
- one `vmatpush.acc.fp8.*` or `vmatpop.fp8.acc.*` of a `1024`-byte `FP8` tile takes
  `16` cycles
- one `vmatpush.acc.bf16.*` or `vmatpop.bf16.acc.*` of a `2048`-byte `BF16` tile takes
  `32` cycles

## Transfer Granularity by Structure

The baseline transfer sizes implied by the local geometry are:

- tensor register: `1024` bytes
- MXU weight slot: `1024` bytes
- MXU accumulation buffer: `2048` bytes

These sizes are derived from:

- one `32 x 32 FP8` tile per tensor register or weight slot
- one `32 x 32 BF16` tile per accumulation buffer

## Initialization and Visibility

The architecture does not require deterministic reset contents for general data storage.

Unless explicitly initialized by software or the host:

- `DRAM` contents are undefined
- `VMEM` contents are undefined

The reference model may instantiate unspecified state deterministically using the frozen
initialization seed and randomization controls from the system-parameter document.
