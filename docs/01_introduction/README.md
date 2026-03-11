# Introduction

This section contains the high-level overview of our design.

## Overview

In modern physical AI systems, Vision-Language-Action (VLA) models have emerged as a dominant paradigm for semantic perception, reasoning, and embodied control. Unlike traditional model-based control pipelines, VLA policies are built upon large transformer architectures that directly map multi-modal inputs to robot actions. These models process high-dimensional visual and language tokens through dense matrix multiplications, resulting in substantial low-precision floating-point compute demand.

However, VLA inference differs fundamentally from datacenter LLM/VLM workloads. Since the model is deployed at edge on-board the robot, the inference is typically single-batch with fixed sequence length, as camera resolution and prompt formats remain constant during deployment. Unlike autoregressive LLM decoding, VLA policies generally operate without causal attention masks and resemble a “prefill-only” execution pattern. This removes token-by-token generation dynamics but emphasizes high-throughput, latency-bounded full-sequence 2D matrix computations.

Together, these characteristics create a unique design point: high arithmetic intensity, deterministic workload structure, strict real-time latency constraints, and aggressive energy efficiency requirements. These factors motivate specialized hardware accelerators optimized specifically for edge-deployed VLA inference rather than repurposed datacenter-oriented architectures.

To this extent, we propose the Physical AI Accelerator (referred as “Accelerator” in the following context). The design is specialized for deterministic, energy-efficient execution of VLA policies.

The proposed architecture is guided by the following principles:
- Energy-efficient low-precision floating-point datapath design.
- Fully deterministic execution within on-chip SRAM boundaries.
- Throughput-matched 2D matrix functional units optimized for dense GEMM, attention, and activation layers in VLA workload.
- Static-scheduled instruction architecture, enabling predictable timing, reduced control overhead, and compiler-managed resource orchestration.


## Design Features

Features of the design include:
- Statically scheduled instruction
  - Latency of the arithmetic functional units are fully deterministic, allowing programs to be optimized during compilation time.
- Long-chime operations.
  - Each instruction launches a multi-cycle operation that sweeps through the matrix tiles.
  - Greatly reduces instruction throughput requirement and allows latency hiding of scalar instructions.
- Branch delay slot.
  - Hides functional unit pipeline latency.
- High throughput matrix execution unit with two accumulation designs.
  - Direct comparison between systolic array units and parallel inner-product trees for area and energy efficiency.
- Low precision floating point MACs.


## System Architecture


Main blocks on the chip include:
- Saturn-V Tile, containing a Rocket/Shuttle RISC-V application core with Saturn RISC-V vector datapath.
- Accelerator Tile, containing the Accelerator described in this document.
- System memory bus interconnect.
- Scratchpad memory that is accessed by both tiles for fast instruction / data memory operation.
- Peripheral devices including UART and JTAG for system debugging.
- Serial TileLink for off-chip DRAM access.

The Accelerator is contained in its own tile, operating in parallel with the Saturn-V Tile. The Saturn core acts as the host processor. It configures the accelerator via MMIO-mapped control and status registers, loads kernel programs into the Accelerator instruction memory, and triggers execution.

Once launched, the Accelerator operates independently. It directly fetches instructions from its local instruction memory and dispatches instructions to the internal functional units. Upon kernel completion, the Accelerator updates its status registers and halts, signaling completion to the Saturn core for further command or task scheduling.


## Accelerator Architecture

Within its tile, the Accelerator employs a static-scheduled instruction architecture designed for deterministic execution and bounded latency. A lightweight control unit fetches instructions from instruction memory and orchestrates all execution units in a deterministic manner.

These functional units include:
- Two Matrix Execution Units (MXU) with different datapath designs
- One Vector Processing Unit (VPU)
- One Cross-lane Transpose Unit (XLU)
- One Direct Memory Access (DMA) Unit
- One Scalar Arithmetic and Logic Unit (SALU)

