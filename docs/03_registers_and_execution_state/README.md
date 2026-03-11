# Register and Execution State

This section defines the architectural states of the Accelerator.

## Control and Status Registers

The Accelerator contains a group of control and status registers used for address calculation, synchronization and blocking, and controlling the execution of the datapath.
The control and status registers are exposed as memory-mapped IO devices on the chip’s system bus. The writable registers can be programmed by the Saturn-V RISC-V core for control purposes, and the readable registers can also be accessed by the Saturn-V core to monitor the execution states.
There are 32 instances of 64-bit general purpose control registers, labeled `x0`~`x31`. These registers store integer types that can be interpreted as either immediate constants or memory address locations.
There is one additional control and status register: the program counter `pc`, which holds the address of the current instruction.

> Design Rationale
>
> The general purpose registers are set to be 64-bit to ensure that they can reference the 16 GiB DRAM data, which is mapped in the range `0x00_8000_0000` to `0x04_0000_0000`. We do not implement vector lane masking, since the workloads are well-dimensioned enough, and we can always zero-fill the extra lanes. The MXU and VPU will perform operations on all elements in the data registers, and DMA and the DRAM program loader will need to implement proper stride and zero-padding to ensure correct element indexing and access.

## List of Control and Status Registers

The CSR space is organized as follows (offsets are from the Accelerator CSR base address; all CSRs are 64-bit unless noted):

### Control (CTRL)

This register is with read-write permission.

The register is located at 0x4010_0000.

Bit 0 ENABLE: When bit is set (1), the accelerator proceeds with program execution. When the bit is cleared (0), the instruction fetch of the accelerator is halted.

### Status (STATUS)

This register is with read-only permission.

The register is located at 0x4010_0008.

### PC (PC)

This register is with read-write permission.

The register is located at 0x4010_0010.

The register contains the current program counter value of the accelerator.

DMA Status (DMA)

This register is with read-only permission.

The register is located at 0x4010_0018.

Bit 0~7 contains the busy status of the DMA channels.

General purpose CSRs (x0 ~ x31)

These registers are with read-write permission.

These registers are located at 0x4010_0100 ~ 0x4010_01FF.

64-bit general purpose control registers used for constants and addresses.


## Tensor Data Registers

Most of the on-chip space is allocated to the data registers, which store the intermediary results for reuse between the matrix and vector functional units.

There are 64 matrix tensor registers (MREG), labeled `m0`~`m63`, and 2 weight buffer registers (WB), labeled `w0` and `w1`. Each tensor register is 2048 bytes, and each weight buffer register is 512 bytes. In total, the matrix tensor register file can hold `64 * 2048 B = 128 kiB` of data, and the weight buffer can hold `2 * 512 B =  1 kiB` of data. 

MREG can be accessed by all the functional units. The read and write ports are multiplexed across functional units.

WB are local to the MXUs and therefore can only be accessed by the MXUs. The values serve as the second operand for matrix multiplication.

MREG is physically banked; the architecture exposes a no-conflict requirement: the program must ensure that simultaneous register accesses issued in the same cycle do not target the same bank.

In case the program creates a bank conflict, the younger instruction takes priority, and older instruction is retired immediately.

For example, given a program:

```assembly
vadd m0, <operand>, delay=0
vadd m32, <operand>
```

The result to `m0` may only be partially written back (only the first few rows are written back), but the results to `m32` are fully written back, overriding the remaining m0 cycles where write bank conflict happens.


### Register Views

Software can access the tensor register in different views with different instructions:

- 16-bit floating point number
- 8-bit floating point number

Both the tensor register and the weight buffer register store 2D matrix data. The weight always stores a FP8 matrix tile of shape (32 x 16).

The exact shape of the data stored in the tensor register depends on the view. Each tile has 64 rows. The exact number of columns of the tile depends on the way we are viewing the register:

- When viewing the registers as 8-bit values, each matrix register stores a tile of shape (64 x 32).

- When viewing the registers as 16-bit values, each matrix register stores a tile of shape (64 x 16).

Multiple views can be used simultaneously on different registers, and each arithmetic instruction specifies its own view to registers by specifying the required data types.

Each cycle, one row, or 32 bytes, of the register can be read out. This means:
- When viewing the registers as 8-bit values, each cycle 32 elements can be read and written per functional unit.
- When viewing the registers as 16-bit values, each cycle 16 elements can be read and written per functional unit.

> Design Rationale
>
> Note the difference of read throughput, in terms of elements, in different register file views. We design the functional unit throughput to match this: Each cycle, MXU consumes twice the number of elements than it generates. 
> 2026-02-23 update: we remove FP32 support. The widest number used in the target workload will be quantized to BF16.

### Reset Behavior

After a reset, the value of the control and status registers are reset to their default values. The values of the data registers are UNKNOWN. The program needs to explicitly load data into the registers to ensure a deterministic value.

### Physical Implementation

The activation registers are implemented as 1-read 1-write ported (1r1w) SRAM bank arrays for high storage density. At full utilization, the activation register needs to provide multiple streams of data operands for MXUs, VPU, and DMA stores each cycle. The maximum read and write bandwidth is set to 1024 bytes per cycle. In the TSMC SRAM library, the widest SRAM macro is up to 256 bits. To fulfill the read bandwidth, we will use 32 instances of SRAM macros in parallel.

In this arrangement, each SRAM bank contains the content of 2 registers. Since each register is 64 deep, the total depth of the SRAM bank would be 128 entries.

The implication of this banking strategy to software programming is that registers within the same bank cannot be used at the same time. For example, when register m1 is used by the MXU, register m33 cannot be used by other functional units.

The weight registers need to sustain high read bandwidth from the MXUs. As a result, they are implemented as flip-flop registers that are distributed near the MXU MAC trees.


> Design Rationale
>
> Because we need to be able to read and write to the same bank at different addresses at the same cycle, we need at least a one-write-port one-read-port (1w1r) SRAM bank. 

### Reading and Writing the Registers

For each read port of the functional units, the functional unit should set its own mux to select one of the 32 SRAM banks to read from. Similarly, for write, each functional unit should control the target destination bank to receive the data. Software will ensure that instructions are scheduled to avoid bank conflict.
