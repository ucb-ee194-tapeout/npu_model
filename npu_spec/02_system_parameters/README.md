# System Parameters

## Frozen Architectural Shape

| Parameter | Value | Meaning |
|---|---:|---|
| `INSN_WIDTH` | `32` bits | Fixed instruction width |
| `INSN_ALIGN` | `4` bytes | Instruction alignment |
| `NUM_XREG` | `32` | Scalar register count |
| `CONTROL_FLOW_DELAY_SLOTS` | `2` | Required branch / jump delay slots |
| `NUM_EREG` | `32` | Scale register count |
| `EREG_BITS` | `8` | Bits per scale register |
| `NUM_MREG` | `64` | Tensor register count |
| `MREG_ROWS` | `32` | Rows per tensor register |
| `MREG_ROW_BYTES` | `32` bytes | Bytes per tensor-register row |
| `MREG_BYTES` | `1024` bytes | Bytes per tensor register |
| `MXU_COUNT` | `2` | Architected MXU count |
| `WEIGHT_SLOTS_PER_MXU` | `2` | Weight slots per MXU |
| `WEIGHT_SLOT_BYTES` | `1024` bytes | Bytes per MXU weight slot |
| `ACCUM_BUFFER_ROWS` | `32` | Rows per MXU accumulation buffer |
| `ACCUM_BUFFER_COLS_BF16` | `32` | BF16 columns per MXU accumulation buffer |
| `ACCUM_BUFFER_BYTES` | `2048` bytes | Bytes per MXU accumulation buffer |
| `MXU0_ARRAY_ROWS` | `32` | Rows in the `mxu0` systolic fabric |
| `MXU0_ARRAY_COLS` | `32` | Columns in the `mxu0` systolic fabric |
| `MXU1_ARRAY_ROWS` | `32` | Rows in the `mxu1` reduction-tree fabric |
| `MXU1_ARRAY_COLS` | `32` | Columns in the `mxu1` reduction-tree fabric |
| `VPU_LANES_BF16` | `16` | BF16 lanes in the baseline VPU |
| `DMA_CHANNELS` | `8` | Architected DMA channels |
| `DMA_ALIGN` | `32` bytes | DMA alignment and granularity |
| `IMEM_BASE` | `0x0002_0000` | IMEM base address |
| `IMEM_SIZE` | `128 KiB` | IMEM capacity |
| `VMEM_BASE` | `0x2000_0000` | VMEM base address |
| `VMEM_SIZE` | `1 MiB` | VMEM capacity |
| `DRAM_BASE` | `0x8000_0000` | DRAM base address |
| `DRAM_SIZE` | `16 GiB` | DRAM capacity |

## Frozen Timing Classes and Bandwidth Fragments

| Parameter | Value | Meaning |
|---|---:|---|
| `MXU0_MATMUL_LATENCY_CYCLES` | `96` | One `mxu0` matmul launch latency class |
| `MXU1_MATMUL_LATENCY_CYCLES` | `35` | One `mxu1` matmul launch latency class |
| `VPU_SIMPLE_OP_LATENCY_CYCLES` | `66` | Pipelineable BF16 VPU latency class (binary, unary, transcendental, `vmov`, `vpack` / `vunpack`) |
| `VPU_NON_PIPELINEABLE_OP_LATENCY_CYCLES` | `130` | Non-pipelineable BF16 VPU latency class (column reductions: `vredsum`, `vredmin`, `vredmax`) |
| `VPU_ROW_REDSUM_LATENCY_CYCLES` | `39` | Row sum reduction latency class (`vredsum.row.bf16`) |
| `VPU_ROW_REDMINMAX_LATENCY_CYCLES` | `34` | Row min/max reduction latency class (`vredmin.row.bf16`, `vredmax.row.bf16`) |
| `VPU_VLI_LATENCY_CYCLES` | `65` | Vector load-immediate latency class (`vli.*`) |
| `XLU_TRANSFORM_LATENCY_CYCLES` | `66` | XLU latency class |
| `OFFCHIP_LINK_WIDTH_BITS` | `32` | DRAM-link beat width |
| `OFFCHIP_LINK_CORE_CYCLES_PER_BEAT` | `2` | Off-chip serialized beat time |
| `DMA_OFFCHIP_COMMAND_WORDS` | `2` | DRAM-side DMA command overhead |
| `VMEM_BUS_WIDTH_BITS` | `256` | VMEM beat width |
| `VMEM_BUS_CORE_CYCLES_PER_BEAT` | `1` | VMEM beat time |
| `VMEM_TENSOR_ALIGN` | `32` bytes | VMEM alignment for tensor-facing transfers |
| `TRACE_TICKS_PER_CYCLE` | `1` | Trace timestamp granularity |

## Software-Model Initialization Parameters

| Parameter | Value | Meaning |
|---|---:|---|
| `INIT_SEED` | `42` | Deterministic pseudo-random initialization seed |
| `RANDOMIZE_DRAM` | `true` | Randomize DRAM at power-on in the Python model |
| `RANDOMIZE_VMEM` | `true` | Randomize VMEM at power-on in the Python model |
| `RANDOMIZE_SCALAR_REGISTERS` | `true` | Randomize scalar registers except `x0` |
| `RANDOMIZE_SCALE_REGISTERS` | `true` | Randomize scale registers |
| `RANDOMIZE_TENSOR_REGISTERS` | `true` | Randomize tensor registers |
| `RANDOMIZE_WEIGHT_SLOTS` | `true` | Randomize MXU weight-slot state |
| `RANDOMIZE_ACCUM_BUFFERS` | `true` | Randomize MXU accumulation-buffer state |
| `RANDOMIZE_DMA_BASE` | `true` | Randomize `dma.base` at power-on in the Python model |

## Derived Whole-Register Views

The frozen tensor-register views are:

- `32 x 32 FP8_E4M3` in one `m` register
- `32 x 16 BF16` in one `m` register
- one full `32 x 32 BF16` tile occupies two consecutive `m` registers
- each BF16 VPU operand or result names the low register of one consecutive
  two-register `32 x 32 BF16` tile

The frozen MXU-local views are:

- one `32 x 32 FP8` weight tile per weight slot
- one `32 x 32 BF16` accumulation tile per accumulation buffer

## Derived Throughput Roofs

At the normalized performance-model level:

- DRAM roof: `2 B/cycle`
- VMEM roof: `32 B/cycle`
- `mxu0` peak: `32 * 32 * 32 * 2 / 96 ≈ 683 FLOPs/cycle`
- `mxu1` peak: `32 * 32 * 32 * 2 / 35 ≈ 1872 FLOPs/cycle`
- total MXU peak: `≈ 2555 FLOPs/cycle`
