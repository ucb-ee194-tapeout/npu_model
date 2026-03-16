# Functional Units

This section outlines the design of the functional units.

## Control Unit (CTRL)

The control unit fetches instructions from the local instruction memory, decodes the instruction as control signals, and dispatches them to the functional units.

### Pipelining

The static scheduling architecture provides benefits on simplifying the pipeline design:

- Control of the pipeline is simplified by eliminating pipeline interlocks.
- Branch latency hidden by branch delay slots. Serial instructions proceed through the pipeline with a fixed relative phase difference between micro-ops.

The pipeline stages are divided into three phases:

- Instruction Fetch (IF)
- Instruction Decode (ID)
- Execute (EX)

#### Instruction Fetch (IF)

The fetch phases of the pipeline are:

- IFG: Instruction address generate and send.
- IFR: Instruction receive.

During the IFG stage, the program address is generated from either the program counter (PC) or the branch target, and is sent to the instruction memory. During IFR, the result instruction is returned from the memory.

#### Instruction Decode (ID)

The decode stage only contain one phase:

- ID: instruction decode and dispatch.

During the ID0 stage, the source registers, destination registers, and associated control signals are decoded for the execution, and the instruction is dispatched to the corresponding functional unit.

#### Execute (EX)

The execution latency is determined by the type of instructions.

> Design Rationale
>
> In an ideal case, to keep all functional units busy, we could employ a VLIW-style architecture as the instruction interface of the control logic. In this scheme, a 2D grid of functional units and time slots are filled with the scheduled instruction operations. However, an issue of this VLIW-style design is that there will be many empty slots. For example, if a single matrix operation lasts for 64 cycles, then there will be 63 empty slots after the matmul instruction for that functional unit. This drastically increases the required instruction memory size and instruction fetch throughput.
> Instead, we serialize the VLIW instruction bundles as serial instructions with delay annotations. Each cycle, we only fetch one instruction. If the instruction is annotated with a delay, we stall for that amount of cycles before dispatching this instruction to the functional units.
> Assuming that most of the instructions are long-latency, we should be able to serialize the program without creating extra overhead.

![](/docs/images/vliw_and_serialized_execution.png)


### Branch Instructions

Branch instructions execute in one EX stage, but due to pipeline latency between IFG and EX0, there are two architectural delay slots after any branch. The two instructions immediately following a branch always execute, regardless of whether the branch is taken.

> Design Rationale
>
> Two delay slots preserve a simple control pipeline while keeping branch behavior deterministic and compiler-schedulable.

### Memory Stalls

The DMA is interfacing with the external memory system, where variations and perturbations happen. Therefore, all memory accesses need to be guarded by a memory barrier instruction. A memory stall occurs when the data is not available after our predicted amount of delays, when program reaches the barrier instruction. The memory stall prevents the next instruction from being dispatched. The in-flight instructions can still continue their execution, draining the execute pipelines.


## Scalar Arithmetic and Logical Unit (SALU)

SALU is a simple scalar integer unit used for address calculation, loop control, predication updates, and CSR manipulation. The SALU implements operations similar to a RISC-V RV64I processor.

The latency of the operations are all single-cycle. Results are readable next cycle unless otherwise specified by instruction latency tables.

> Design Rationale
>
> Without SALU, the programming model cannot cleanly express address generation and control without pushing complexity to the host CPU. SALU is intentionally minimal to preserve static scheduling and small area.


## Matrix Execution Unit (MXU)

This section outlines the structure of the two matrix multiply units. MXU0 is implemented as a systolic array, and MXU1 is implemented as parallel inner-product trees. Both units are designed to operate in a weight-stationary dataflow (weight tile is loaded and reused over multiple cycles). Weight-stationary dataflow is selected since it provides a better balance between the required read and write bandwidth between the PE and the register file.
In weight-stationary dataflow, weight tiles are loaded into the two architectural weight buffers w0 and w1 (each 32×16 FP8). During MXU operation, one WB entry is “active” (feeding the compute array/trees) while the other may be filled by DMA, enabling architectural double buffering.

> Design rationale
>
> For weight buffer, restricting WB to two entries keeps area small while still enabling overlap of compute with weight fetch. The explicit dma_wait points ensure correctness when a kernel depends on fresh weights.
>
> For MXU datapath design, the main design goal here is to minimize the energy consumption of the matrix unit. These two matrix units reflect the two approaches to perform the accumulation of the matrix multiplication:
> - In-place accumulation. The multiply result is added to the previous partial result by a 2:1 adder.
> - Reduction. Multiple multiply results are added together via an adder tree. This is commonly used to perform dot products.
>
> The tradeoff between the two accumulation designs is at the data width and number of copies of the adders and the accumulation register. For the in-place accumulation design, we need to expand the mantissa for a small amount to account for the overflow/underflow of adding two numbers; whereas for the reduction tree we need to keep a larger amount of extra bits to account for this multi-number addition. However, for a N2 throughput PE array, the in-place adder and accumulation register structure needs to be duplicated N2 times; but a reduction tree design only requires N copies of the tree adders and accumulation registers. Note that the total number of adders required in both designs are the same (or at least similar), and the difference lies in the bit width required in each adder. We use these two matrix units to serve as a point of comparison to compare and contract their characteristics.

### Systolic Array (MXU0)

This matrix unit design implements a 32 x 16 dimension systolic array with weight-stationary dataflow. Each tile in the array is capable of computing one low-precision floating point multiplication and one accumulation per cycle. Since the systolic array operates in weight-stationary dataflow, the summed result is propagated to the next cell. This MXU is therefore capable of performing 32 x 16 = 512 multiply-accumulate (MAC) operation, giving us a throughput of 1024 FLOPs/cycle.


### Parallel Inner-product Tree (MXU1)

This matrix unit design consists of 16 inner-product reduction trees. Each tree is capable of computing 32 pairs of low-precision floating point multiplication and sum the results as a single scalar value per cycle. This MXU is therefore capable of performing 16 x 32 = 512 MAC operation, giving us a throughput of 1024 FLOPs/cycle.

The MXU operates on a weight-stationary dataflow schedule. The weights values are loaded to the weight registers and are reused for multiple cycles.

Before initiating a matrix multiply instruction, the weight needs to be loaded to the weight buffer registers located near each tree. There are two sources to load the weight:
- Loading from DRAM (`dma.load.mxu0.ch<N> {w0-3}, <address>, <size>`). In this case, DMA will be configured to load a chunk of weight data from DRAM directly to the weight registers. The matrix multiplication can be started after the DMA barrier instruction has been satisfied.
- Loading from the activation register (`vmov.m.w {w0-3}, <vsrc>`). In this case, a special move instruction is used to transfer weight data from one of the activation registers. Since the latency is known, no barrier needs to be used. The matrix multiplication instruction can be dispatched after a fixed amount of delay. The first 16 rows of the source matrix register will be moved to the weight buffer.

The content in the accumulation register can be reset to zero, for the case when a new matrix multiplication is started; or loaded from the matrix register, for the case where we keep accumulating the partial product from a previous operation.
During each cycle of a matrix multiply instruction, a tile of the activation operand with shape (2, 32) will be loaded from the matrix register file to the tree, computing a partial sum of shape (2, 16). This partial sum will then be immediately moved out from the accumulation buffer back to one of the matrix registers. This operation continues for 32 cycles, fully consuming the content within one matrix register.


## Vector Processing Unit (VPU)

The vector processing unit handles all the floating point operations other than GEMM. These operations include:

- Addition

  vd = vs1 + vs2

- Subtraction

  vd = vs1 - vs2

- Multiplication

  vd = vs1 * vs2

- Reciprocal

  vd = 1 / vs

- Square root

  vd = sqrt(vs)

- Log of 2

  vd = log2(vs)

- Exponential of 2

  vd = 2 ^ vs

- Natural exponential

  vd  = e ^ vs

- Sine

  vd = sin(vs)

- Cosine

  vd = cos(vs)

- Hyperbolic tangent

  vd = tanh(vs)


## Cross-lane Transpose Unit (XLU)

The cross-lane transpose unit handles matrix transpose operation.
The core of the XLU is a 16 x 64 systolic shift unit. When pushing data into the XLU, the instruction specifies a source tensor register. The BF16 elements from that source register are then shifted into the XLU rightward for 64 cycles. The rightmost 16 columns of the shift unit also contain a downward path. When popping the data elements out of the XLU, the rightmost 16x16 tile will be moved downwards and loaded to the first 16 rows of the destination register. To fully compute a transpose of the source (16 x 64) matrix, 4 register and 4 pop instructions are required.


## Direct Memory Access Load-Store Unit (DMA)

The direct memory access load-store unit is in charge of handling the memory transfer between the off-chip memory and either the matrix register file or the weight buffer. Each DMA operation transfers a tile of 2D matrix. The data will be tiled to appropriate tiled layouts in memory by the compiler, so DMA only needs to implement unit-stride contiguous transfer.

After a DMA instruction is issued, the DMA unit will generate multiple requests and send them out to the memory bus.

### Terminology

We define an operation to be a single load/store instruction intended to operate on all data in a single matrix register. We define a transaction to be a single ready-valid handshake on the bus on which the DMA is attached. A transaction issues a single cycle TileLink request, whose logical transfer size is equal to the width of the bus (i.e. beatBytes). The downstream, or the memory-side managers will respond one TileLink response per TileLink request in order (TODO: check). 

### Architecture

The DMA should expose 8 operation interfaces to a program. This allows up to 8 concurrent operations (not transactions) to be issued at the same time. Each interface has a busy flag, raised on the cycle the after operation-issuing operation performs its handshake, and lowered on the cycle after the last response transaction corresponding to the operation is received.

The latency of any DMA operation should be assumed to be variable; therefore, any instructions involving a data hazard with a matrix register currently used by the DMA must be fenced. Although DMA operations are intended to be served in order (FCFS), the actual order at which they complete should not be assumed either.

Transfer size: for both load and store, a transfer size in bytes will also be provided. The operation always starts from the first element in the register, and operates on the data in range [0, size-1]. The size will always be divisible by row size, which means the size will be row granularity, and  there will not be partial rows.

Load instructions specify an interface ID, and take a 64-bit source DRAM address, a destination matrix register name or weight buffer entry, and a transfer size (`dma.load.ch<N> <vdest>, <address>, <size>`, and `dma.loadweight.ch<N> <vdest>, <address>, <size>`). The DMA should retrieve the contents at the DRAM address, assumed to be bus-aligned, in unit stride until the number of bytes required to fill exactly one matrix register has been requested. The operation should happen asynchronously, meaning the control flow is free to move on after the single-cyle issue.

Store instructions specify an interface ID, and take a 32-bit destination DRAM address, a source matrix register name, and a transfer size (`dma.store.ch<N> <vsource>, <address>, <size>`). The DMA should retrieve all contents of the matrix register and store to the DRAM address in unit stride. Similar to the load instruction, the store instruction is also asynchronous.
Fence instructions specify the target interface (`dma.wait.ch<N>`). When the busy flag for the given interface is high, the DMA should stall the control flow until the given interface lowers its busy flag. If the busy flag is already low at instruction issue time, the DMA should immediately return a response. Up to one fence can be in progress (by design).

### Microarchitecture

Towards the core front end, the DMA should expose a valid-only input interface for instruction issue. It should also expose a valid-only output interface for fence response.

Towards the downstream memory, the DMA should expose a single TileLink client with 6 bits source ID (for up to 64 in-flight memory transactions). Other parameters of the client will be diplomatically negotiated downstream.

The DMA should keep an 8-entry operation queue that stores received load and store instructions. This queue should enqueue and dequeue operations in core issue order. Note that fence instructions are not queued; they should be kept track of independently. The queue is dequeued from when the corresponding operation has just received its final response transaction. It’s software’s responsibility to ensure the queue does not overflow, which can be guaranteed if all interface IDs are fenced before being reused.

We define a request head and a response head to the queue. The request head points to the current active operation, denoting the current operation for which requests are being issued. Once all requests have been issued for an operation, the request head moves to the next item in the queue. Note that this means not all responses need to be received for the DMA to move onto another operation. Once all responses have been received for an operation, the response head moves on.

We keep track of 3 states: the request transaction counter, the response transaction counter, and the in-flight counter. Both the request and response counters will wrap when they respectively reach the number of transactions required to serve a matrix register.

Upon reset, all counters are set to 0. When the request head points at an invalid entry, the request counter is kept at 0 (it should wrap for any head change by design). Similarly, the response counter is kept at 0 when the response head is invalid. Any time the DMA TileLink interface performs a handshake (fire) on the A channel, the request counter is incremented; ditto for the D channel/response counter. The A channel should stay valid until the request counter wraps or maximum in-flights are reached (more below); the D channel should stay always ready. 

All TileLink requests will carry a source ID. It’s important that no in-flight source IDs are reused for compliance; as a result, there is a limit to the maximum number of TileLink request transactions that are in flight (defined to be requests for which a response hasn’t yet been received) equal to the number of possible source IDs. Given a 6-bit source ID, up to 64 requests can be in flight. We therefore track this limit using the in-flight counter. This counter is incremented when A fires, decremented when D fires, or kept as-is when both fires (or both don’t). While we can simply cycle through 0~63 for each request regardless of operation or in-flights, we must deassert request valid when the in-flight counter reaches 64 (inclusive - 7 bit counter required).
