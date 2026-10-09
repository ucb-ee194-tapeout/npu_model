# Functional assembly (.fs)

The functional interpreter executes instruction semantics without modeling
hardware cycles, resource conflicts, physical register limits, DMA channels, or
delay slots. It is intended for compiler-facing program validation. The
existing timed assembler and simulator remain the path for true assembly
(`.es`) and performance evaluation.

## Virtual operands

Register operands use a type prefix and a programmer-chosen name:

| Register class | `.fs` spelling | Example |
| --- | --- | --- |
| Scalar | `x.<name>` | `x.address` |
| Exponent | `e.<name>` | `e.scale` |
| Matrix | `m.<name>` | `m.tile` |
| Weight buffer | `w.<name>` | `w.weights` |
| Accumulator | `acc.<name>` | `acc.partial` |

Names are virtual identifiers, not encoded register indices. The interpreter
assigns internal IDs and stores values in sparse, unbounded register and bank
files. MXU0 and MXU1 have separate functional weight-buffer and accumulator
state, matching their distinct architectural storage. Functional execution
does not allocate those names to hardware registers or banks.
Virtual registers, matrix backing slots, weight buffers, accumulators, VMEM,
and DRAM begin zero-filled, matching the model's deterministic reset state.

Assembly operand order and punctuation follow the existing instruction
patterns. For example:

```asm
addi x.left, x.zero, 9
addi x.right, x.zero, 7
add x.total, x.left, x.right
```

Branch and jump labels are supported. Functional control flow has no hardware
delay slot: a taken branch continues at its target. Programs finish when the
PC moves past the final instruction. The interpreter stops with an error if
control flow leaves the program range or exceeds the configured instruction
limit.

## Functional-only differences

`delay` and `dma.wait.ch<N>` are not accepted. DMA channel numbers are omitted:

```asm
dma.config x.dram_base
dma.load x.vmem_address, x.dram_offset, x.byte_count
dma.store x.dram_offset, x.vmem_address, x.byte_count
```

DMA transfers complete atomically as copies. Channel-free functional DMA
operations share the true ISA's data movement semantics; no overlap or transfer
latency is simulated. Cycle-dependent CSR operations and halt instructions are
outside the initial functional subset.

Scratchpad placement is explicit. The programmer supplies addresses and
tracks layout and lifetime. Scalar load/store addresses are byte addresses;
`vload`/`vstore` use the true ISA's word-based base and immediate convention.
Memory accesses are bounds checked. VMEM and DRAM start zero-filled unless the
caller initializes them through the Python API.

## Running and using the interpreter

```bash
uv run scripts/run_functional.py kernel.fs
uv run scripts/run_functional.py kernel.fs --numerics rtl
```

The Python API accepts initial memory images without requiring large dense
DRAM allocations:

```python
from npu_model.functional import FunctionalInterpreter, load_functional_assembly

program = load_functional_assembly("kernel.fs")
model = FunctionalInterpreter()
result = model.run(program, dram={0: bytes([1, 2, 3, 4])})
print(result.named_registers(program))
```

To inspect the true assembly syntax and encoding information derived from the
existing ISA definitions, export the compiler-facing manifest:

```bash
uv run scripts/export_isa.py --output atlas-isa.json
```

The JSON schema is versioned. It includes ordered source operands, nested
operand syntax such as `imm(x(rs1))`, types and register/immediate bounds,
operand access roles, encoding formats and opcode/funct selectors, and the
functional mnemonic set. It intentionally contains no timing or resource
scheduling information.
