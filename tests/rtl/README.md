# Scalar RTL trace regressions

`scalar_cases.json` contains assembly and the exact instruction words supplied to
the RTL. `scalar_traces.json` records the real Chisel `ScalarCore` simulated with
Verilator. The Python test compares fetch PC, executing PC, instruction issue,
halt, redirect timing, and scalar CSR output data on every cycle.

The harness is `src/test/scala/atlas/scalar/NpuModelScalarTraceTest.scala`, relative
to the accelerator repository. It uses a synchronous instruction ROM, idle
engines, an explicitly driven DMA channel busy signal, and a one-cycle memory
response after the scalar LSU command. The response contains `0x12345678`.
These traces validate the scalar frontend and its command/response registers;
they do not validate the complete DMA, vector engines, or TileLink IMEM path.

Cycle 1 is the first cycle following the host start edge. Signals are sampled
before the edge, except `next_pc`, which is sampled after it. An instruction
fetched in cycle 1 executes in cycle 2. CSR write data is sampled before scalar
load writeback, which exposes the RTL's lack of load-result bypass.

From the accelerator repository, with Java and Verilator on `PATH`, regenerate:

```sh
./mill --no-server atlas.test.testOnly atlas.scalar.NpuModelScalarTraceTest
cd npu-model
python -m pytest tests/test_rtl_scalar_traces.py
```

`ATLAS_SCALAR_TRACE_INPUT` and `ATLAS_SCALAR_TRACE_OUTPUT` optionally override
the fixture paths. When changing cases, update their instruction words with
the assembler before regenerating the trace.
