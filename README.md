# NPU-Model

This is a preliminary, experimental performance model for various NPU architecture.
Notice that this is neither performant or accurate in any sense.
This model makes highly ideal assumption, and is only useful for educational purposes or early stage exploration.
Use at your own risk.

This perf model is tick-based.


### Tick Based vs Event Based
These two terms primarily discerns the execution model of the simulator.

A tick-based performance model specifies hardware behavior per cycle.
At every cycle, every FU is "ticked" one by one.
This is easier to write but usually slower.

An event based performance model specifies hardware behavior as events.
Classically, there exists a global event queue.
Producers insert events into this queue to invoke its consumers.
Consumers are invoked when their invocation is poped from the queue.
This is harder to write and reason about, but can be faster if done right.

We choose to implement tick-based for its simplicity.

This is execution driven.

### Execution Driven vs Trace Driven
These two terms primarily discerns the frontend behavior of the simulator.

A trace-driven performance model does not require simulating any architectural value.
A trace is a predefined, recorded sequence of instructions and memory access addresses.
Assuming microarchitectural features does not change the ordering and content of the trace, (which may or may not be true), replaying these traces stimulates the uarch.
This allows for faster simulation, and is usually easier to write as the correctness check is loose.

An execution-driven performance model requires simulating all architectural value.
Instead of a trace, its frontend fetch decision is dependent on the actual architectural value.
This is slower and harder to write, and it is easy diverge from realistic execution behaviors.

We choose to do execution driven regardless, because the NPU ISA encodes static delays to avoid dynamic dependency checking.
Hence a functional model for it necessinates simulating timing.
So writing this model in execution-driven fashion achives both goals in one shot.


## Hardware Modeling

The hardware model uses a **tick-based simulation** approach with reverse pipeline order ticking to properly propagate values:

- **Pipeline Stages**:
  - **IFU (Instruction Fetch Unit)**: Fetches instructions from program memory
  - **IDU (Instruction Decode Unit)**: Decodes and dispatches instructions to execution units
  - **EXUs (Execution Units)**: Execute instructions with configurable latencies

- **Execution Unit Types**:
  - **ScalarExecutionUnit**: Single-cycle scalar operations (add, sub, branches)
  - **MatrixExecutionUnit**: Multi-cycle matrix operations (matmul)
  - **DmaExecutionUnit**: Memory transfer operations with flag-based synchronization

- **Architectural State**:
  - Scalar register file (XRF)
  - Matrix register file (MRF) with configurable dimensions
  - Memory subsystem
  - Program counter (PC) management

## Software Modeling

Programs are represented as sequences of `Instruction` objects:

- **Instruction Set Architecture (ISA)**: Decorator-based instruction definitions
- **Dynamic Instruction Instances (Uops)**: Runtime instruction tracking with unique IDs
- **Configurable Programs**: Easy creation of test programs and benchmarks
- **Memory Initialization**: Support for pre-loaded memory regions

## Trace Generation

The simulator generates detailed execution traces for visualization:

- **Perfetto-compatible output**: View traces at https://ui.perfetto.dev
- **Per-instruction tracking**: Fetch, Decode, Execute, Retire stages
- **Lane-based visualization**: Separate lanes for each execution unit
- **Cycle-accurate timing**: Precise cycle-by-cycle execution flow


## Usage

### Requirements

- Python >= 3.10
- Dependencies managed via `uv` (see `pyproject.toml`)

### Basic Simulation

Run a simulation with default configuration:

```bash
uv run ./scripts/run.py --program MatmulProgram --hardware_config DefaultHardwareConfig -o matmul.json
```

### Custom Configuration

Specify hardware configuration and program:

```bash
uv run scripts/run.py --hardware_config DefaultHardwareConfig -p AddiProgram -o trace.json
```

### Command-Line Options

- `--hardware_config`: Hardware configuration class (default: `DefaultHardwareConfig`)
- `-p, --program`: Program to execute (default: `AddiProgram`)
- `-o, --output`: Output trace file (default: `trace.json`)
- `--max-cycles`: Maximum simulation cycles (default: `1000`)

### Viewing Traces

1. Run simulation to generate trace file
2. Open https://ui.perfetto.dev
3. Load the generated trace file (`.json`)
4. Explore cycle-by-cycle execution flow


## Architecture

### Pipeline Flow

```
IFU → IDU → EXUs (Scalar/Matrix/DMA)
 ↓     ↓      ↓
Fetch Decode Execute → Retire
```

The simulator ticks in **reverse pipeline order** to ensure proper data propagation:
1. Execution units claim instructions from IDU outputs
2. IDU claims instructions from IFU and dispatches to EXUs
3. IFU fetches new instructions (if not stalled)
4. Cycle counter advances

### Claim-Based Handshaking

Pipeline stages use `StageData` with claim-based handshaking:
- Downstream stages **claim** data from upstream stages
- Upstream stages **stall** if their data isn't claimed
- Prevents data loss and ensures proper backpressure


## Project Structure

```
npu_model/
├── model_npu/
│   ├── configs/           # Hardware and program configurations
│   │   ├── hardware/      # Hardware configuration classes
│   │   ├── programs/      # Example programs
│   │   └── isa_definition.py  # ISA instruction definitions
│   ├── hardware/          # Hardware components
│   │   ├── core.py        # Core orchestration
│   │   ├── ifu.py         # Instruction Fetch Unit
│   │   ├── idu.py         # Instruction Decode Unit
│   │   ├── exu.py         # Base Execution Unit + Scalar EXU
│   │   ├── mxu.py         # Matrix Execution Unit
│   │   ├── dma.py         # DMA Execution Unit
│   │   ├── arch_state.py  # Architectural state
│   │   └── config.py      # Hardware configuration base
│   ├── software/          # Software representation
│   │   ├── instruction.py # Instruction and Uop classes
│   │   └── program.py     # Program class
│   ├── logging/           # Trace logging
│   │   └── logger.py      # Perfetto trace logger
│   ├── isa.py             # ISA framework
│   └── simulation.py      # Simulation orchestration
├── scripts/
│   └── run.py             # Main simulation runner
└── pyproject.toml         # Project dependencies
```


## Creating Custom Programs

Define a new program by subclassing `Program`:

```python
from model_npu.software import Instruction, Program

class MyProgram(Program):
    instructions = [
        Instruction(mnemonic="addi", args={"rd": 1, "rs1": 0, "imm": 5}),
        Instruction(mnemonic="addi", args={"rd": 2, "rs1": 1, "imm": 3}),
        Instruction(mnemonic="nop", args={}),
    ]
```


## Creating Custom Hardware Configurations

Define custom hardware by subclassing `HardwareConfig`:

```python
from model_npu.hardware.config import HardwareConfig
from model_npu.isa import IsaSpec

class MyHardwareConfig(HardwareConfig):
    name = "MyNPU"
    fetch_width = 1
    isa = IsaSpec
    matrix_shape = (32, 32)  # 32x32 matrices
    memory_size = 32 * 32 * 4 * 4
    execution_units = {
        "Scalar0": "ScalarExecutionUnit",
        "Matrix0": "MatrixExecutionUnit",
    }
```

### Pre-commit

Run pre-commit checks:

```bash
uv run pre-commit run --all-files
```

### Adding New Instructions

1. Define instruction effect in `model_npu/configs/isa_definition.py`:

```python
@instr("my_instr", instruction_type=InstructionType.SCALAR)
def my_instr(state: ArchState, args: Dict[str, int]) -> None:
    # Implement instruction behavior
    state.set_xrf(args["rd"], ...)
```

2. Use in programs:

```python
Instruction(mnemonic="my_instr", args={"rd": 1, ...})
```
