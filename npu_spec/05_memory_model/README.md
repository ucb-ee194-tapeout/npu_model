# Memory Model

## High-Level Rule

The baseline memory system keeps the asynchronous boundary narrow:

- `IMEM` fetch is local and deterministic
- `VMEM` is the sole on-chip tensor staging memory
- `DRAM` access is off-chip and asynchronous
- DMA is the only `DRAM <-> VMEM` path

## On-Chip Transfers

The RTL uses software-scheduled, independently progressing scalar, VLOAD, and
VSTORE paths. These operations do not block the frontend. Scheduling violations
assert. For an S1 issue at cycle T, scalar stores write at T+1, scalar loads write
back at T+3, and vector transfers write rows at T+3 through T+34.

Scalar LSU addresses are bytes. VLOAD/VSTORE bases are word addresses and their
12-bit immediate contributes 32 words per unit. The resulting line address is
`(x[rs1] + (sign_extend(imm12) << 5)) >> 3`, matching ScalarCore's wiring.
VMEM comprises six contiguous 256 KiB banks.

## Asynchronous DMA

DMA is channelized and asynchronous.

Each channel supports:

- at most one outstanding transfer
- independent busy / completion state
- `dma.wait.chN` synchronization

The microarchitecture may implement the DMA channels with shared internal data paths or
arbitration, provided the architecture-visible channel behavior is preserved.

`dma.wait.chN` behaves as a frontend fence:

- neither instruction allocates a normal execute-stage slot while it is holding decode
- if channel `N` is already done when `dma.wait.chN` reaches decode, the instruction
  spends that cycle in decode and retires directly
- if channel `N` is not yet done, decode remains occupied until the transfer completes,
  then the instruction retires directly from decode
- younger instructions shall not issue past that decode fence until the wait retires

`delay N` issues and retires immediately, then holds the following instruction
in S1 for N cycles. Halt/illegal detection precedes this stall in the RTL.

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

## DMA Engine Timing

The DMA engine moves data in `DMA_ALIGN` (32-byte) beats over a TileLink port.
Its timing is structural, not a closed form:

- a transfer issued at `T` occupies one of `DMA_CHANNELS` command slots from
  `T+1`; its channel reads busy from `T+1` until the cycle after the slot retires
- beats of the oldest undispatched slot issue one per cycle while fewer than
  `64` beats are outstanding and the memory accepts; later slots' beats issue
  while earlier slots' responses are still returning
- a store reads VMEM one line per cycle ahead of its beats, subject to the bank
  grant; a line read at `C` can issue at `C+2`
- load data is written to VMEM in the response cycle if the bank grants it;
  LSU accesses to a bank deny the DMA that bank, and a DMA write beats a DMA read
- a slot retires in the first cycle in which every beat has issued and none is
  outstanding; slots may retire out of issue order
- the VMEM operand is a 32-bit word address; the line is its bits `[18:3]`
  (low bits ignored, higher bits wrap) and the engine asserts if the line range
  leaves VMEM
- the off-chip address is `{dma.base, x[rs]}`, 64 bits; DRAM occupies
  `0x8000_0000` upward (64 GiB in the simulation target)

The time a beat spends beyond the port is an environment parameter. The
reference target is VCS simulation of EE290SimConfig, where the port reaches
DRAMSim2 (DDR3, Chipyard's `+dramsim` default) through TileLink buffers and a
64-bit AXI port. The reference model's default memory backend is a
bandwidth-latency curve (after the Mess simulator) measured on that path with
DRAM refresh disabled: a lone-beat round trip of `46` cycles per Get and `42`
cycles per Put, and a streaming rate of about `23` cycles per `32`-byte beat
shared by loads and stores. Refresh adds a start-time-dependent offset of up
to about `130` cycles per transfer window that the model does not reproduce.

### Legacy baseline estimate

The previous frozen baseline charged each transfer a closed-form latency, which
the reference model keeps as `dma_transfer_cycles` for reference only:

- `OFFCHIP_BYTES_PER_BEAT = OFFCHIP_LINK_WIDTH_BITS / 8 = 4`
- `VMEM_BYTES_PER_BEAT = VMEM_BUS_WIDTH_BITS / 8 = 32`
- `dma_offchip_cycles(bytes) = ceil((bytes + 4 * DMA_OFFCHIP_COMMAND_WORDS) / OFFCHIP_BYTES_PER_BEAT) * OFFCHIP_LINK_CORE_CYCLES_PER_BEAT`
- `vmem_transfer_cycles(bytes) = ceil(bytes / VMEM_BYTES_PER_BEAT) * VMEM_BUS_CORE_CYCLES_PER_BEAT`
- `dma_transfer_cycles(bytes) = max(dma_offchip_cycles(bytes), vmem_transfer_cycles(bytes))`

The reference model's `fixed` backend can still derive link parameters from
these constants (one beat every `16` cycles after `4` cycles of latency) for
directed tests; it is not the default.

For the frozen baseline values:

- one off-chip beat costs `2` core cycles
- one VMEM beat costs `1` core cycle
- one `vload` or `vstore` of a `1024`-byte tensor register takes `34` cycles
- one `vmatpush.acc.fp8.*` or `vmatpop.fp8.acc.*` of a `1024`-byte `FP8` tile takes
  `32` cycles
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
