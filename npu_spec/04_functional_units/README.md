# Instruction Set

The Accelerator is programmed using a static scheduled architecture with no runtime data dependency checking. All functional units are therefore scheduled explicitly by software.

This kind of architecture is designed from the characteristic that the target workload exhibits:
- Coarse-grained (very large data blocks) data-parallel computation.
- High degrees of predictable instruction-level parallelism.
- Relatively simple and streamlined control flow.

These properties enable efficient static scheduling by the compiler while reducing hardware complexity associated with dynamic scheduling, hazard detection, and issue control.

## Delay and Latency

The execution of floating-point instructions can be defined in terms of delay slots and functional unit latency. The number of delay slots is equivalent to the number of additional cycles required after the source operands are read for the result to be available for reading. For a single-cycle type instruction, operands are read on cycle i and produce a result that can be read on cycle i + 1. For a 4-cycle instruction, operands are read on cycle i and produce a result that can be read on cycle i + 4. Delay slots are equivalent to execution or result latency.

## List of Instructions

To keep a single consistent implementation, please refer to the functional and performance model for the instruction definitions:

https://github.com/ucb-ee194-tapeout/npu_model/blob/main/model_npu/configs/isa_definition.py


## Instruction Encoding

Each instruction is encoded as a 32-bit word.
