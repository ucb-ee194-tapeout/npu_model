# Introduction

## Overview

The Tapeout NPU is a statically scheduled accelerator-oriented machine with a scalar
control path and a tile-oriented tensor datapath.

The baseline machine contains:

- one scalar integer register file of `32` registers, `x0` through `x31`
- one tensor register file of `64` registers, `m0` through `m63`
- one scale register file of `32` registers, `e0` through `e31`
- two architecturally visible matrix execution units, `mxu0` and `mxu1`
- one vector processing unit, `vpu`
- one tensor transform / reduction unit, `xlu`
- one instruction memory, `IMEM`
- one on-chip tensor / vector memory, `VMEM`
- one off-chip backing memory, `DRAM`
- eight architected DMA channels between `DRAM` and `VMEM`

## Design Principles

The baseline is guided by the following principles:

- static scheduling rather than dynamic dependency checking
- single-issue in-order fetch / decode / issue
- deterministic on-chip timing for a fixed configuration
- explicit overlap of long-chime functional units
- one narrow asynchronous boundary at `DRAM <-> VMEM`
- simple software-model, compiler, and RTL alignment

## High-Level Execution Model

The frontend is intentionally narrow:

- one fixed-width `32`-bit instruction stream
- one fetch stream
- at most one new instruction issued per cycle

Execution is overlap-oriented rather than superscalar:

- long-chime tensor instructions may remain active for multiple cycles after issue
- younger instructions may still issue to other available units
- issue stalls only on structural conflicts and architecturally defined blocking
  conditions
- architectural state updates become visible when the producing instruction completes

Control flow is explicit and compiler-visible:

- branches and jumps have exactly `2` architecturally visible delay slots
- a control-transfer instruction appearing in a delay slot is illegal

## Baseline Tile Geometry

The local baseline uses `32 x 32` architectural tensor tiles.

This affects all major tensor-facing structures:

- MXU arithmetic operates on `32 x 32` tiles
- each tensor register stores one `32 x 32 FP8` tile or one `32 x 16 BF16` half-tile
- each MXU accumulation buffer stores one `32 x 32 BF16` tile
- the baseline VPU operates on a `32 x 16 BF16` whole-register view using `16 BF16`
  lanes

## Memory-System Shape

The architecture exposes three disjoint memory regions:

- `IMEM` at `0x0002_0000`, used for instruction fetch
- `VMEM` at `0x2000_0000`, used for on-chip tensor and scalar data access
- `DRAM` at `0x8000_0000`, used for off-chip backing storage

The intended architectural dataflow is:

- scalar loads and stores access `VMEM` only
- tensor `vload` and `vstore` access `VMEM` only
- DMA is the only architected `DRAM <-> VMEM` path
- all other tensor motion is explicit on-chip movement between register files and MXU
  local state
