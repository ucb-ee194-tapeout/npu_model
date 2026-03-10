# NPU Model Interface Notes

This note maps the standalone MLIR pipeline in `compiler/` to the simulator-facing
`model_npu` program model, with focus on matmul/ukernel paths.

## Current MLIR Pipeline

- `linalg.generic` (matmul-like) / `linalg.matmul`
- `-> npu_kernel.ukernel_generic` (or `npu_kernel.matmul`)
- `-> npu_schedule.ukernel_launch` (or `npu_schedule.matmul_tile`)
- `-> npu_isa.*` skeleton ops
- `-> textual ISA` via `npu-translate`

The FP8 matmul ukernel test is:

- `compiler/test/lowering/linalg_generic_fp8_ukernel_pipeline.mlir`

## How `model_npu` Programs Are Represented

Programs are defined as a Python `Program` with:

- `instructions: List[Instruction]` where each instruction has
  `mnemonic` and `args` (`dict[str, int]`)
- `memory_regions: List[(base_addr, tensor)]`

Relevant files:

- `model_npu/configs/programs/gemma_attention.py`
- `model_npu/configs/programs/gemma_mlp.py`
- `model_npu/workload/gemma_blocks.py`
- `model_npu/configs/isa_definition.py`

## Instruction-Level Mapping

Directly compatible with our current emitted text:

- `npu_isa.dma_load` -> `dma.load rd=..., base=..., size=..., flag=...`
- `npu_isa.dma_wait` -> `dma.wait flag=...`
- `npu_isa.matmul_mxu0` -> `matmul.mxu0 rd=..., rs1=..., rs2=...`

Used in Gemma programs but not yet modeled in our ISA dialect:

- `dma.load.mxu1`
- `dma.store`
- vector compute chain:
  `vmul`, `vexp`, `vreduce.sum`, `vrcp`, (and others in rms_norm)

## Important Consistency Check

In `model_npu/configs/isa_definition.py`, `matmul.mxu0` reads weight from
`mxu1` WB (`read_wb_fp8("mxu1", rs2)`), while some legacy test programs still
show `dma.load.mxu0` before `matmul.mxu0`.

Before extending the lowering, confirm intended hardware contract:

1. Should `matmul.mxu0` consume `mxu0` WB or `mxu1` WB?
2. Which program set is the source of truth (Gemma configs vs. older matmul test)?

## Recommended Next Dialect Steps (for Gemma parity)

1. Add ISA ops:
   - `npu_isa.dma_load_mxu1`
   - `npu_isa.dma_store`
   - `npu_isa.vmul`, `npu_isa.vexp`, `npu_isa.vreduce_sum`, `npu_isa.vrcp`
2. Add a schedule-level softmax fragment op (or explicit vector chain op group).
3. Lower `npu_schedule.ukernel_launch` by ukernel symbol:
   - `npu_uk_matmul_*` -> current matmul skeleton
   - `npu_uk_gemma_mlp_*` -> matmul + vmul + store pattern
   - `npu_uk_gemma_attention_*` -> matmul + softmax vector chain + matmul + store
4. Keep text emitter format identical to `Instruction.__str__` style so output can
   be consumed directly by simulator loaders.

## Status In This Tree

Implemented:

1. ISA ops added:
   - `npu_isa.dma_load_mxu1`
   - `npu_isa.dma_store`
   - `npu_isa.vmul`
   - `npu_isa.vexp`
   - `npu_isa.vreduce_sum`
   - `npu_isa.vrcp`
2. Schedule op added:
   - `npu_schedule.softmax_fragment`
3. `npu_schedule.ukernel_launch` symbol-based lowering:
   - `npu_uk_matmul_*` -> matmul skeleton
   - `npu_uk_gemma_mlp_*` -> `dma.load.mxu1` + matmul + `vmul` + `dma.store`
   - `npu_uk_gemma_attention_*` -> matmul + softmax vector chain + matmul + `dma.store`
4. Text ISA translation emits `mnemonic key=value, ...` matching
   `Instruction.__str__`.

Note on flags:

- Current lowering uses DMA flags in range `[0, 2]` to stay compatible with
  `DefaultHardwareConfig` (three flags in simulator state).

## Validation Loop

For each new ukernel symbol:

1. Run `npu-opt` with `--mlir-print-ir-after-all`.
2. Emit text ISA with `npu-translate`.
3. Compare mnemonic/arg schema against `model_npu/configs/isa_definition.py`.
4. Execute with `compiler/scripts/run_simulator_smoke.py`.

Concrete artifacts in this repo:

- `compiler/test/lowering/linalg_generic_fp8_ukernel_pipeline.mlir`
- `compiler/test/lowering/ukernel_symbol_to_isa.mlir`
- `compiler/test/outputs/matmul_fp8_print_after_all.txt`
- `compiler/test/outputs/gemma_attention_print_after_all.txt`
