# Tapeout NPU Spec

This document set defines the current architecture and microarchitecture for the
Tapeout NPU.

- MXU PE sizes are `32 x 32`
- one tensor register stores one `32 x 32 FP8` tile
- one tensor register stores one `32 x 16 BF16` half-tile
- one full `32 x 32 BF16` tile occupies two consecutive tensor registers
- the baseline VPU width is `16 BF16` lanes

These documents are intended to be read together:

- `00_preface`: scope and document map
- `01_introduction`: high-level machine overview and design principles
- `02_system_parameters`: frozen shared parameters
- `03_registers_and_execution_state`: architectural state, memory map, alignment, and reset rules
- `04_functional_units`: frontend and execution-unit implementation direction
- `05_memory_model`: memory-system behavior, blocking rules, DMA semantics, and transfer timing
- `06_instruction_set`: encoding framework and ISA contract

This spec set supersedes the older split between informal architecture notes and
standalone architecture/microarchitecture documents. The existing `npu_spec/00-06`
layout is now the canonical baseline.
