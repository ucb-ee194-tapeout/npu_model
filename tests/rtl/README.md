# RTL comparison fixtures

These are outputs of **actual Chisel RTL simulated with Verilator**, not traces
produced by the Python model. Canonical Scala harnesses are in [scala/](scala/).
The regeneration script temporarily installs them in the accelerator's Mill
test-source tree, runs the simulations, then removes its temporary directory.
No production RTL is modified.

From `npu-model`, with Java and Verilator on `PATH`:

```sh
python3 scripts/regenerate_rtl_fixtures.py --chipyard CHIPYARD
python3 scripts/regenerate_rtl_fixtures.py --check
python3 -m pytest tests/test_rtl_*traces.py tests/test_rtl_arithmetic.py tests/test_rtl_artifacts.py
```

Regeneration also recreates the exhaustive unary BF16 tables in
`npu_model/configs/data/`. It takes several minutes, primarily compiling the
full 32×32 MXUs. `provenance.json` has one record for each harness suite: the
fingerprints of its fixtures, and one fingerprint of the RTL and harness sources
that recorded them. Python tests detect stale fixtures when the adjacent RTL changes.
The standalone Python checkout checks the fixture fingerprints without requiring RTL.
`--only TEXT` runs only the suites whose names contain `TEXT` (case-sensitive; no
match is an error). The other suites keep their records, so a stale suite stays
stale until it runs again.

| Fixture | Real RTL instantiated | Signals compared / coverage |
| --- | --- | --- |
| `scalar_cases.json`, `scalar_traces.json` | `ScalarCore` | Fetch/S1 PCs, issue, halt, illegal, CSR write data; 16 branch, jump, delay, load and DMA-wait scenarios |
| `vector_traces.json` | `VectorEngineTop` | Every MREG read/write cycle, address and output bit; 30 operations, special encodings, overlap, mirrored reads, in-place move and last-write handoff; `.vx` and `.vr` forms; issue-busy of each form |
| `ipt_traces.json` | `InnerProductTreesTop` | Every MREG read/write cycle/address/data; all ten commands, accumulator chaining and overlapping weight push/compute |
| `memory_traces.json` | `LSU`, `XluEngine` | Concurrent scalar/VLOAD/VSTORE and transpose requests, write data, scalar writeback; VMEM and MREG accesses |
| `dma_traces.json` | `DmaEngine`, `Vmem` | Per-cycle TileLink A/D fires, VMEM read/write requests and grants, channel busy; six interleaved load/store commands, out-of-order responses, A-channel backpressure, LSU bank denials |
| `arithmetic.json` | `AnchorAccumulationTree` | 512 seeded random/boundary cases, all 256 FP8 input encodings and FP32 accumulator edge cases |
| `configs/data/*.bin.gz` (under `npu_model/`) | Actual `VectorEngineTop` lane boxes | All 65,536 BF16 encodings for each of 11 unary operations |

Memory-facing harnesses supply synchronous one-cycle responses. They test the
engines and their command/response pipeline, without instantiating TileLink or
a complete AtlasTile. The DMA harness is the exception: it instantiates the
actual `DmaEngine` and `Vmem` with a scripted TileLink responder (per-beat
latency `5 + 2 * (source % 4)`, `tl.a.ready` low every fifth cycle) and
scripted LSU bank traffic. Because `DMA.scala` and `VMEM.scala` depend on
rocket-chip, that harness is compiled by Chipyard's sbt project `sp26atlas`
rather than the Mill `atlas` module; the regeneration script installs it under
`src/test/scala/` (outside the Mill tree, which sbt excludes) and runs
`testOnly atlas.dma.NpuModelDmaTraceTest` from the Chipyard root. Use
`--only dma` to regenerate it alone. The scalar harness uses a synchronous instruction ROM,
idle tensor engines, an explicitly driven DMA busy signal and a fixed scalar
memory response (`0x12345678`). IMEM host arbitration and CSR counters have
separate Python tests, rather than full-system RTL traces.

Signals are sampled before the clock edge; `next_pc` in the scalar trace is
sampled after it. `fetch_pc` and `next_pc` are IMEM word indices; `s1_pc` is the
byte address of the S1 instruction (ScalarCore's illegal-PC output). Scalar
cycle 1 follows the host start edge: the first fetched instruction executes in
cycle 2. Engine cycle 1 is command issue. The memory harness includes
ScalarCore's scalar command/response registers.

Unary table files contain gzip-compressed little-endian uint16 results indexed
by the raw BF16 input encoding. They deliberately retain the RTL's approximation,
NaN, infinity, signed-zero and subnormal behavior. They are model datapath data;
the trace fixtures separately verify indexing, scheduling and writeback.

## Chipyard for the DMA suite

The DMA suite runs in Chipyard's sbt project `sp26atlas`, which is
`generators/atlas-npu`, so that directory of the Chipyard must be this checkout. Give the
Chipyard root with `--chipyard DIR` or `CY_DIR`. The default is two directories
above this checkout, as in `chipyard/generators/atlas-npu`.

If the generator directory of your Chipyard is another checkout, make a shadow
tree. The shadow tree does not change the Chipyard checkout:

```sh
scripts/chipyard_shadow.sh CHIPYARD SHADOW
export COURSIER_CACHE=SHADOW/.coursier PATH=CHIPYARD/.conda-env/bin:$PATH
python3 scripts/regenerate_rtl_fixtures.py --chipyard SHADOW
```

The shadow tree has copies of `build.sbt`, the sbt files of `project/`
(`build.properties` and `*.sbt`) and `scripts/sbt-launch.jar`, and links to the other
Chipyard directories. It adds `project/ShadowTargets.scala`, an sbt plugin that puts
every `target/` into the shadow tree. Its `generators/atlas-npu` is a link
to this checkout. sbt writes its outputs (`target/`) and its caches (`.sbt`,
`.ivy2`, and Coursier with `COURSIER_CACHE`) in the shadow tree. Chipyard's sbt
1.8.2 needs a Java that its Scala 2.12.17 can read, for example Java 20 of the
Chipyard conda environment. Java 21 does not load the build.

With the same environment, the shadow tree also compiles the Chipyard
configurations of `chipyard/config` with the other Chipyard generators:

```sh
cd SHADOW && java -jar scripts/sbt-launch.jar -Dsbt.ivy.home=$PWD/.ivy2 \
  -Dsbt.global.base=$PWD/.sbt -Dsbt.boot.directory=$PWD/.sbt/boot/ chipyard/compile
```
