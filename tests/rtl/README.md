# RTL comparison fixtures

These are outputs of **actual Chisel RTL simulated with Verilator**, not traces
produced by the Python model. Canonical Scala harnesses are in [scala/](scala/).
The regeneration script temporarily installs them in the accelerator's Mill
test-source tree, runs the simulations, then removes its temporary directory.
No production RTL is modified.

From `npu-model`, with Java and Verilator on `PATH`:

```sh
python scripts/regenerate_rtl_fixtures.py
python scripts/regenerate_rtl_fixtures.py --check
python -m pytest tests/test_rtl_*traces.py tests/test_rtl_arithmetic.py tests/test_rtl_artifacts.py
```

Regeneration also recreates the exhaustive unary BF16 tables in
`npu_model/hardware/data/`. It takes several minutes, primarily compiling the
full 32×32 MXUs. `provenance.json` fingerprints the RTL sources, harnesses and
artifacts; Python tests detect stale fixtures when the adjacent RTL changes.
The standalone Python checkout checks artifact hashes without requiring RTL.

| Fixture | Real RTL instantiated | Signals compared / coverage |
| --- | --- | --- |
| `scalar_cases.json`, `scalar_traces.json` | `ScalarCore` | Fetch/S1 PCs, issue, halt, illegal, CSR write data; 14 branch, jump, delay, load and DMA-wait scenarios |
| `vector_traces.json` | `VectorEngineTop` | Every MREG read/write cycle, address and output bit; 29 operations, special encodings, overlap, mirrored reads, in-place move and last-write handoff |
| `sa_traces.json`, `ipt_traces.json` | `SystolicArrayTop`, `InnerProductTreesTop` | Every MREG read/write cycle/address/data; all seven commands, accumulator chaining and overlapping weight push/compute |
| `memory_traces.json` | `LSU`, `XluEngine` | Concurrent scalar/VLOAD/VSTORE and transpose requests, write data, scalar writeback; VMEM and MREG accesses |
| `arithmetic.json` | `E4M3FMA`, `AnchorAccumulationTree`, `FPUtils` converters | 512 seeded random/boundary cases, all 256 scale encodings and FP8 input encodings |
| `hardware/data/*.bin.gz` (under `npu_model/`) | Actual `VectorEngineTop` lane boxes | All 65,536 BF16 encodings for each of 11 unary operations |

Memory-facing harnesses supply synchronous one-cycle responses. They test the
engines and their command/response pipeline, without instantiating TileLink or
a complete AtlasTile. The scalar harness uses a synchronous instruction ROM,
idle tensor engines, an explicitly driven DMA busy signal and a fixed scalar
memory response (`0x12345678`). IMEM host arbitration and CSR counters have
separate Python tests, rather than full-system RTL traces.

Signals are sampled before the clock edge; `next_pc` in the scalar trace is
sampled after it. Scalar cycle 1 follows the host start edge: the first fetched
instruction executes in cycle 2. Engine cycle 1 is command issue. The memory
harness includes ScalarCore's scalar command/response registers.

Unary table files contain gzip-compressed little-endian uint16 results indexed
by the raw BF16 input encoding. They deliberately retain the RTL's approximation,
NaN, infinity, signed-zero and subnormal behavior. They are model datapath data;
the trace fixtures separately verify indexing, scheduling and writeback.
